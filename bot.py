import logging
import requests
import time
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
from telegram.constants import ParseMode
import re
from ratelimit import limits, sleep_and_retry
from cachetools import LRUCache
import backoff
import json
import unicodedata

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
TELEGRAM_TOKEN = "7888772385:AAEGpPDVxsFBqjskcI__taTaa01VveBrVPM"
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = [
    "mistralai/devstral-small:free",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "agentica-org/deepcoder-14b-preview:free"
]
DEFAULT_MODEL = MODELS[0]
MAX_MESSAGE_LENGTH = 4096  # Максимальная длина сообщения в Telegram
RATE_LIMIT_PERIOD = 120  # 2 минуты
RATE_LIMIT_CALLS = 2  # 2 запроса в 2 минуты
CACHE_SIZE = 100  # Размер кэша для ответов
MAX_HISTORY = 50  # 50 сообщений в истории диалога
MAX_RETRIES = 3  # Максимум попыток для экспоненциального отката
REQUEST_TIMEOUT = 15  # Таймаут запросов в секундах

# Инициализация кэша и сессии
cache = LRUCache(maxsize=CACHE_SIZE)
session = requests.Session()

# Экранирование специальных символов для MarkdownV2
def escape_markdown_v2(text):
    """Экранирует все специальные символы для Telegram MarkdownV2."""
    if not isinstance(text, str):
        text = str(text)
    special_chars = r'([_\*\[\]\(\)~`>\#\+\-=|\{\}\.!])'
    return re.sub(special_chars, r'\\\1', text)

# Проверка на китайские иероглифы
def has_chinese(text):
    """Проверяет, содержит ли текст китайские иероглифы (U+4E00–U+9FFF)."""
    for char in text:
        if 0x4E00 <= ord(char) <= 0x9FFF:
            return True
    return False

# Очистка текста
def clean_text(text, is_code=False):
    """Очищает текст от не-UTF-8 символов и проверяет на китайские иероглифы."""
    try:
        # Нормализация Unicode
        text = unicodedata.normalize('NFKC', text)
        # Удаляем не-UTF-8 символы
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        # Пропускаем проверку на китайские символы для кода
        if is_code:
            return text
        # Проверяем на китайские иероглифы
        if has_chinese(text):
            logger.warning(f"Chinese characters detected in response: {text[:50]}...")
            return "Ошибка: Ответ содержит текст на китайском языке. Пожалуйста, повторите запрос."
        return text
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text

# Разбиение длинного сообщения
def split_message(text, max_length=MAX_MESSAGE_LENGTH):
    """Разбивает текст на части, не превышающие max_length."""
    if len(text) <= max_length:
        return [text]
    
    messages = []
    while text:
        if len(text) <= max_length:
            messages.append(text)
            break
        split_pos = text[:max_length].rfind('\n')
        if split_pos == -1:
            split_pos = text[:max_length].rfind(' ')
        if split_pos == -1:
            split_pos = max_length
        messages.append(text[:split_pos])
        text = text[split_pos:].lstrip()
    return messages

# Форматирование кода для Telegram
def format_code_block(code, language=""):
    """Форматирует код в MarkdownV2 для Telegram."""
    cleaned_code = clean_text(code, is_code=True)
    if "Ошибка: Ответ содержит текст на китайском языке" in cleaned_code:
        return cleaned_code
    escaped_code = escape_markdown_v2(cleaned_code)
    return f"```{language}\n{escaped_code}\n```"

# Проверка статуса API-ключа
async def check_key():
    """Проверяет статус API-ключа OpenRouter."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "TelegramBot/1.0",
        "HTTP-Referer": "https://your-bot-domain.com",
        "X-Title": "TelegramOpenRouterBot"
    }
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "system", "content": "Отвечай на русском языке"}, {"role": "user", "content": "Тест"}]
    }
    
    logger.info("Checking OpenRouter API key status...")
    try:
        response = session.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        remaining_requests = response.headers.get('X-RateLimit-Remaining', 'Unknown')
        reset_time = response.headers.get('X-RateLimit-Reset', 'Unknown')
        logger.info(f"API key check successful. Remaining requests: {remaining_requests}, Reset time: {reset_time}")
        return f"API-ключ активен! Осталось запросов: {remaining_requests}, Сброс лимита: {reset_time}"
    except requests.exceptions.HTTPError as e:
        retry_after = e.response.headers.get('Retry-After', 'Unknown') if e.response else 'Unknown'
        logger.error(f"API key check failed: {e}, Retry-After: {retry_after}")
        return f"Ошибка API: {str(e)}, Retry-After: {retry_after}"
    except requests.exceptions.RequestException as e:
        logger.error(f"API key check network error: {e}")
        return "Ошибка сети при проверке API-ключа."

# Команда /check_key
async def check_key_command(update: Update, context: CallbackContext):
    """Проверяет статус API-ключа OpenRouter."""
    result = await check_key()
    await update.message.reply_text(result)

# Команда /status
async def status(update: Update, context: CallbackContext):
    """Проверяет текущие лимиты OpenRouter API."""
    result = await check_key()
    await update.message.reply_text(result)

# Запрос к OpenRouter API
@sleep_and_retry
@limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
@backoff.on_exception(backoff.expo, requests.exceptions.HTTPError, max_tries=MAX_RETRIES, giveup=lambda e: e.response is None or e.response.status_code != 429)
async def query_openrouter(message, model=DEFAULT_MODEL, history=None):
    """Отправляет запрос к OpenRouter API с кэшированием и обработкой Retry-After."""
    cache_key = f"{model}:{message}:{json.dumps(history)}"
    if cache_key in cache:
        logger.info("Returning cached response")
        return cache[cache_key]
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "TelegramBot/1.0",
        "HTTP-Referer": "https://your-bot-domain.com",
        "X-Title": "TelegramOpenRouterBot"
    }
    
    system_message = {
        "role": "system",
        "content": "Отвечай исключительно на русском языке. Не используй китайский, английский или другие языки, если только пользователь явно не попросит. Для кода используй Python, если не указано иное."
    }
    full_history = [system_message] + (history or [{"role": "user", "content": message}])
    
    payload = {
        "model": model,
        "messages": full_history
    }
    
    logger.info(f"Sending request to OpenRouter, model: {model}, message: {message[:50]}...")
    try:
        response = session.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        result = clean_text(data['choices'][0]['message']['content'])
        cache[cache_key] = result  # Сохраняем в кэш
        remaining_requests = response.headers.get('X-RateLimit-Remaining', 'Unknown')
        reset_time = response.headers.get('X-RateLimit-Reset', 'Unknown')
        logger.info(f"API request successful, model: {model}, response: {result[:50]}..., Remaining requests: {remaining_requests}, Reset time: {reset_time}")
        return result
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 429:
            retry_after = e.response.headers.get('Retry-After', '60')
            try:
                retry_after = int(retry_after)
            except ValueError:
                retry_after = 60
            logger.warning(f"Rate limit hit, waiting {retry_after} seconds")
            await asyncio.sleep(retry_after)
            raise
        logger.error(f"HTTP error from OpenRouter: {e}")
        return f"Ошибка API: {str(e)}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {e}")
        return "Ошибка сети. Попробуйте позже."
    except (KeyError, IndexError) as e:
        logger.error(f"Error parsing OpenRouter response: {e}")
        return "Ошибка обработки ответа от API."
    finally:
        logger.info("Request to OpenRouter completed")

# Обработчик команды /start
async def start(update: Update, context: CallbackContext):
    """Отправляет приветственное сообщение и очищает историю."""
    context.user_data['history'] = []  # Очищаем историю
    context.user_data['model'] = DEFAULT_MODEL
    await update.message.reply_text(
        "Привет! Я чат-бот, использующий OpenRouter API. Отправь мне сообщение, и я отвечу на русском!\n"
        f"Текущая модель: {DEFAULT_MODEL}\n"
        "Используй /models для выбора модели через кнопки.\n"
        "Диалог сохраняется (до 50 сообщений).\n"
        "Используй /clear для очистки истории диалога.\n"
        "Используй /check_key для проверки API-ключа.\n"
        "Используй /status для проверки лимитов API."
    )

# Обработчик команды /clear
async def clear(update: Update, context: CallbackContext):
    """Очищает историю диалога."""
    context.user_data['history'] = []
    await update.message.reply_text("История диалога очищена.")

# Обработчик команды /models
async def models(update: Update, context: CallbackContext):
    """Показывает инлайн-кнопки для выбора модели."""
    keyboard = [
        [InlineKeyboardButton(model, callback_data=f"model:{model}")] for model in MODELS
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Выберите модель:",
        reply_markup=reply_markup
    )

# Обработчик выбора модели через кнопки
async def button_callback(update: Update, context: CallbackContext):
    """Обрабатывает выбор модели через инлайн-кнопки."""
    query = update.callback_query
    await query.answer()
    model = query.data.split("model:")[1]
    if model in MODELS:
        context.user_data['model'] = model
        await query.message.reply_text(f"Модель изменена на: {model}")
    else:
        await query.message.reply_text(f"Модель не найдена. Доступные модели: {', '.join(MODELS)}")

# Обработчик текстовых сообщений
async def handle_message(update: Update, context: CallbackContext):
    """Обрабатывает входящие текстовые сообщения."""
    user_message = clean_text(update.message.text, is_code=False)
    model = context.user_data.get('model', DEFAULT_MODEL)
    
    logger.info(f"Processing user message: {user_message[:50]}...")
    # Получаем или инициализируем историю
    history = context.user_data.get('history', [])
    
    # Добавляем сообщение пользователя в историю
    history.append({"role": "user", "content": user_message})
    
    # Ограничиваем историю
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    
    # Запрос к OpenRouter
    response = await query_openrouter(user_message, model, history)
    
    # Добавляем ответ бота в историю
    history.append({"role": "assistant", "content": response})
    context.user_data['history'] = history  # Сохраняем историю
    
    # Проверяем, содержит ли ответ код
    if '```' in response:
        parts = response.split('```')
        formatted_response = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Текст вне кода
                formatted_response += escape_markdown_v2(clean_text(part, is_code=False))
            else:  # Код
                lines = part.split('\n', 1)
                language = lines[0].strip() if len(lines) > 1 and lines[0].strip() else ""
                code = lines[1] if len(lines) > 1 else lines[0]
                formatted_response += format_code_block(code, language)
    else:
        formatted_response = escape_markdown_v2(clean_text(response, is_code=False))
    
    # Разбиваем сообщение
    messages = split_message(formatted_response)
    
    # Отправляем каждую часть
    for msg in messages:
        try:
            await update.message.reply_text(
                msg,
                parse_mode=ParseMode.MARKDOWN_V2
            )
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            await update.message.reply_text(
                "Ошибка при отправке сообщения. Попробуйте снова."
            )
    logger.info(f"Message processed and sent: {user_message[:50]}")

# Обработчик ошибок
async def error_handler(update: Update, context: CallbackContext):
    """Обрабатывает ошибки бота."""
    logger.error(f"Update {update} caused error {context.error}")
    if update and update.message:
        await update.message.reply_text(
            "Произошла ошибка. Попробуйте снова позже."
        )

def main():
    """Запускает бота."""
    logger.info("Starting bot...")
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear))
    application.add_handler(CommandHandler("check_key", check_key_command))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("models", models))
    application.add_handler(CallbackQueryHandler(button_callback, pattern="^model:"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    
    # Запуск бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
