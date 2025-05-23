import time
from typing import Dict
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
import httpx
from tenacity import retry, stop_after_attempt, wait_fixed
from OpenRouterChat.config import OPENROUTER_API_KEY, API_TIMEOUT, MODELS, logger
from OpenRouterChat.utils import clean_text, escape_markdown_v2, format_code_message, is_code, sanitize_input, send_message
from cachetools import TTLCache

# Кэш для ответов (TTL 1 час, до 1000 записей)
cache = TTLCache(maxsize=1000, ttl=3600)

def log_event(event_type: str, user_id: int, message: str, exc_info: bool = False) -> None:
    """
    Логирует событие с информацией о пользователе.

    Args:
        event_type (str): Тип события.
        user_id (int): ID пользователя.
        message (str): Сообщение для логирования.
        exc_info (bool): Включать ли информацию об исключении.
    """
    log_message = f"[{event_type}] Пользователь {user_id}: {message}"
    if exc_info:
        logger.exception(log_message)
    else:
        logger.info(log_message)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик команды /start.

    Args:
        update (Update): Объект обновления Telegram.
        context (ContextTypes.DEFAULT_TYPE): Контекст обработчика.
    """
    user_id = update.effective_user.id
    log_event("Команда start", user_id, "Пользователь инициировал команду start")
    keyboard = [
        [InlineKeyboardButton(model["name"], callback_data=f"model:{key}")]
        for key, model in MODELS.items()
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите модель:", reply_markup=reply_markup)

async def callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик выбора модели.

    Args:
        update (Update): Объект обновления Telegram.
        context (ContextTypes.DEFAULT_TYPE): Контекст обработчика.
    """
    query = update.callback_query
    user_id = query.from_user.id
    model = query.data.split(":")[1]
    context.user_data["model"] = model
    cache.clear()  # Очистка кэша при смене модели
    log_event("Смена модели", user_id, f"Новая модель: {model}")
    await query.answer()
    await query.message.edit_text(f"Выбрана модель: {MODELS[model]['name']}")

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def make_api_request(client: httpx.AsyncClient, model_id: str, message_text: str) -> Dict:
    """
    Выполняет запрос к API OpenRouter с повторными попытками.

    Args:
        client (httpx.AsyncClient): HTTP-клиент.
        model_id (str): ID модели.
        message_text (str): Текст запроса.

    Returns:
        Dict: Ответ API.
    """
    response = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": f"Отвечай строго на русском, с правильной грамматикой и пунктуацией, без текста на других языках. Форматируй код в блоках ```python. {message_text}"
            }]
        },
        timeout=API_TIMEOUT
    )
    response.raise_for_status()
    return response.json()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик текстовых сообщений.

    Args:
        update (Update): Объект обновления Telegram.
        context (ContextTypes.DEFAULT_TYPE): Контекст обработчика.
    """
    start_time = time.time()
    user_id = update.effective_user.id
    message_text = sanitize_input(update.message.text)
    model = context.user_data.get("model", "deepcoder")
    model_id = MODELS.get(model, MODELS["deepcoder"])["id"]
    model_name = MODELS.get(model, MODELS["deepcoder"])["name"]

    log_event("Новый запрос", user_id, f"Модель: {model_name}\nСообщение: '{message_text}'")

    # Проверка кэша
    cache_key = f"{model_id}:{message_text.lower().strip()}"
    if any(keyword in message_text.lower() for keyword in ["код", "while", "for"]):
        cached_response = cache.get(cache_key)
        if cached_response:
            log_event("Попадание в кэш", user_id, f"Возвращен кэшированный ответ для: {message_text}")
            await send_message(update, cached_response, ParseMode.MARKDOWN_V2, user_id)
            log_event("Ответ отправлен", user_id, f"Общее время: {time.time() - start_time:.3f}с")
            return

    try:
        async with httpx.AsyncClient(
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Accept-Encoding": "gzip, deflate"}
        ) as client:
            response_data = await make_api_request(client, model_id, message_text)
            reply_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not reply_text:
                log_event("Ошибка API", user_id, "Пустой ответ от OpenRouter")
                await send_message(update, "Не удалось получить ответ от API. Попробуйте позже.", user_id=user_id)
                return

            log_event("Успех API", user_id, f"Длина ответа: {len(reply_text)} символов")

        # Форматирование ответа
        if any(keyword in message_text.lower() for keyword in ["код", "while", "for"]) or "```" in reply_text:
            code_match = re.search(r"```(?:python)?\n?([\s\S]*?)\n?```", reply_text)
            if code_match:
                code = code_match.group(1).strip()
                explanation = re.sub(r"```(?:python)?\n?[\s\S]*?\n?```", "", reply_text).strip()
            else:
                code = reply_text.strip()
                explanation = "Пример кода." if is_code(code) else ""
            formatted_message = format_code_message(code, explanation)
        else:
            formatted_message = escape_markdown_v2(clean_text(reply_text))

        if not formatted_message:
            log_event("Ошибка форматирования", user_id, "Пустое отформатированное сообщение")
            await send_message(update, "Не удалось обработать ответ. Попробуйте уточнить запрос.", user_id=user_id)
            return

        # Сохранение в кэш
        if any(keyword in message_text.lower() for keyword in ["код", "while", "for"]):
            cache[cache_key] = formatted_message

        await send_message(update, formatted_message, ParseMode.MARKDOWN_V2, user_id)
        log_event("Ответ отправлен", user_id, f"Общее время: {time.time() - start_time:.3f}с")

    except httpx.RequestError as e:
        error_msg = f"Ошибка сети при запросе к API: {str(e)}"
        log_event("Ошибка сети", user_id, error_msg, exc_info=True)
        await send_message(update, "Ошибка сети. Попробуйте позже.", user_id=user_id)
    except httpx.HTTPStatusError as e:
        error_msg = f"Ошибка API ❌ Код {e.response.status_code}: {e.response.text}"
        log_event("Ошибка API", user_id, error_msg, exc_info=True)
        await send_message(update, error_msg, user_id=user_id)
    except Exception as e:
        error_msg = f"Непредвиденная ошибка: {str(e)}"
        log_event("Ошибка", user_id, error_msg, exc_info=True)
        await send_message(update, "Произошла непредвиденная ошибка. Попробуйте позже.", user_id=user_id)
