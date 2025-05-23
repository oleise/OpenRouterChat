import logging
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
from telegram.constants import ParseMode
import re
from ratelimit import limits, sleep_and_retry
from cachetools import LRUCache
import backoff
import time
import json

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
RATE_LIMIT_PERIOD = 60  # Период в секундах (1 минута)
RATE_LIMIT_CALLS = 5  # Уменьшено до 5 запросов в минуту для безопасности
CACHE_SIZE = 100  # Размер кэша для ответов
MAX_HISTORY = 10  # Максимум 10 сообщений в истории диалога

# Инициализация кэша и сессии
cache = LRUCache(maxsize=CACHE_SIZE)
session = requests.Session()

# Экранирование специальных символов для MarkdownV2
def escape_markdown_v2(text):
    """Экранирует специальные символы для Telegram MarkdownV2."""
    special_chars = r'([_\*\[\]\(\)~`>\#\+\-=|\{\}\.!])'
    return re.sub(special_chars, r'\\\1', text)

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
    escaped_code = escape_markdown_v2(code)
    return f"```{language}\n{escaped_code}\n```"

# Запрос к OpenRouter API с ограничением частоты и повторными попытками
@sleep_and_retry
@limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
async def query_openrouter(message, model=DEFAULT_MODEL, history=None):
    """Отправляет запрос к OpenRouter API с кэшированием и возвращает ответ."""
    cache_key = f"{model}:{message}:{json.dumps(history)}"
    if cache_key in cache:
        logger.info("Returning cached response")
        return cache[cache_key]
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "TelegramBot/1.0",
        "HTTP-Referer": "https://your-bot-domain.com"  # Замените на ваш домен или оставьте как есть
    }
    payload = {
        "model": model,
        "messages": history or [{"role": "user", "content": message}]
    }
    
    try:
        response = session.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        result = data['choices'][0]['message']['content']
        cache[cache_key] = result  # Сохраняем в кэш
        logger.info(f"API request successful, model: {model}, response: {result[:50]}...")
        return result
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error from OpenRouter: {e}")
        return f"Ошибка API: {str(e)}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {e}")
        return "Ошибка сети. Попробуйте позже."
    except (KeyError, IndexError) as e:
        logger.error(f"Error parsing OpenRouter response: {e}")
        return "Ошибка обработки ответа от API."

# Обработчик команды /start
async def start(update: Update, context: CallbackContext):
    """Отправляет приветственное сообщение и очищает историю."""
    context.user_data['history'] = []  # Очищаем историю при старте
    context.user_data['model'] = DEFAULT_MODEL
    await update.message.reply_text(
        "Привет! Я чат-бот, использующий OpenRouter API. Отправь мне сообщение, и я отвечу!\n"
        f"Текущая модель: {DEFAULT_MODEL}\n"
        "Используй /models для выбора модели через кнопки.\n"
        "Диалог сохраняется (до 10 сообщений)."
    )

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
    user_message = update.message.text
    model = context.user_data.get('model', DEFAULT_MODEL)
    
    # Получаем или инициализируем историю
    history = context.user_data.get('history', [])
    
    # Добавляем сообщение пользователя в историю
    history.append({"role": "user", "content": user_message})
    
    # Ограничиваем историю до MAX_HISTORY сообщений
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    
    # Запрос к OpenRouter с историей
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
                formatted_response += escape_markdown_v2(part)
            else:  # Код
                lines = part.split('\n', 1)
                language = lines[0].strip() if len(lines) > 1 and lines[0].strip() else ""
                code = lines[1] if len(lines) > 1 else lines[0]
                formatted_response += format_code_block(code, language)
    else:
        formatted_response = escape_markdown_v2(response)
    
    # Разбиваем сообщение, если оно слишком длинное
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
    # Задержка перед первым запросом для предотвращения холодного старта
    time.sleep(2)
    
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("models", models))
    application.add_handler(CallbackQueryHandler(button_callback, pattern="^model:"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    
    # Запуск бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
