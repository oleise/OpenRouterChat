import logging
import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import httpx
import re

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Список специальных символов Telegram для MarkdownV2
TELEGRAM_MARKDOWN_SPECIAL_CHARS = r'([_\*\[\]\(\)~`>\#\+\-\=\|\{\}\.\!])'

def escape_markdown_v2(text: str) -> str:
    """
    Экранирует специальные символы для Telegram MarkdownV2.
    """
    return re.sub(TELEGRAM_MARKDOWN_SPECIAL_CHARS, r'\\\1', text)

def log_event(event_type: str, user_id: int, message: str, exc_info: bool = False):
    """
    Логирует событие с информацией о пользователе и, при необходимости, об исключении.
    """
    log_message = f"[{event_type}] User {user_id}: {message}"
    if exc_info:
        logger.exception(log_message)
    else:
        logger.info(log_message)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
    user_id = update.effective_user.id
    log_event("Start command", user_id, "User initiated start command")
    keyboard = [
        [telegram.InlineKeyboardButton("deepcoder", callback_data="model:deepcoder")],
        [telegram.InlineKeyboardButton("mistral-small", callback_data="model:mistral-small")],
        # Другие модели...
    ]
    reply_markup = telegram.InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите модель:", reply_markup=reply_markup)

async def callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик выбора модели."""
    query = update.callback_query
    user_id = query.from_user.id
    model = query.data.split(":")[1]
    log_event("Model changed", user_id, f"New model: {model}")
    await query.answer()
    await query.message.edit_text(f"Выбрана модель: {model}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений."""
    user_id = update.effective_user.id
    message_text = update.message.text
    model = context.user_data.get("model", "deepcoder")  # Модель по умолчанию
    model_id = {
        "deepcoder": "agentica-org/deepcoder-14b-preview:free",
        "mistral-small": "mistralai/devstral-small:free",
        # Другие модели...
    }.get(model, "agentica-org/deepcoder-14b-preview:free")

    log_event("New request", user_id, f"Model: {model_id}\nMessage: '{message_text}'")

    try:
        # Запрос к OpenRouter API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": message_text}]
                },
                headers={"Authorization": "Bearer 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"}  # Убедитесь, что ключ настроен
            )
            response.raise_for_status()
            response_data = response.json()
            reply_text = response_data["choices"][0]["message"]["content"]
            log_event("API success", user_id, f"Response length: {len(reply_text)} chars")

        # Попытка отправить с MarkdownV2
        try:
            escaped_text = escape_markdown_v2(reply_text)
            await update.message.reply_text(
                escaped_text,
                parse_mode=telegram.constants.ParseMode.MARKDOWN_V2
            )
        except telegram.error.BadRequest as e:
            log_event("Markdown error", user_id, "Falling back to plain text")
            # Откат на обычный текст
            await update.message.reply_text(reply_text)

    except httpx.HTTPStatusError as e:
        error_msg = f"API error ❌ Ошибка {e.response.status_code}: {e.response.text}"
        log_event("API error", user_id, error_msg)
        await update.message.reply_text(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        log_event("Error", user_id, error_msg, exc_info=True)
        await update.message.reply_text("Произошла ошибка, попробуйте позже.")

def main():
    """Запуск бота."""
    log_event("BOT STARTED", 0, "=== BOT STARTED ===")
    app = Application.builder().token("7888772385:AAEGpPDVxsFBqjskcI__taTaa01VveBrVPM").build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(callback_query, pattern="model:.*"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    log_event("Polling started", 0, "Polling started")
    app.run_polling()

if __name__ == "__main__":
    main()
