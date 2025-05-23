import logging
import os
import signal
import re
import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import httpx

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Список специальных символов для Telegram MarkdownV2
TELEGRAM_MARKDOWN_SPECIAL_CHARS = r'([_\*\[\]\(\)~`>\#\+\-\=\|\{\}\.\!])'
MAX_MESSAGE_LENGTH = 4096  # Ограничение длины сообщения в Telegram

def escape_markdown_v2(text: str) -> str:
    """Экранирует специальные символы для Telegram MarkdownV2, кроме блоков кода."""
    return re.sub(TELEGRAM_MARKDOWN_SPECIAL_CHARS, r'\\\1', text)

def format_code_message(code: str, explanation: str = "") -> str:
    """
    Форматирует сообщение с кодом и пояснением для Telegram MarkdownV2.
    Код отправляется как копируемый блок, пояснение — как текст.
    """
    # Код не экранируется, так как находится внутри ```python ... ```
    code_block = f"```python\n{code}\n```"
    if explanation:
        # Экранируем только пояснение
        escaped_explanation = escape_markdown_v2(explanation)
        return f"{code_block}\n\n{escaped_explanation}"
    return code_block

def split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list:
    """Разбивает длинное сообщение на части, не превышающие max_length."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def log_event(event_type: str, user_id: int, message: str, exc_info: bool = False):
    """Логирует событие с информацией о пользователе и, при необходимости, об исключении."""
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
        [InlineKeyboardButton("deepcoder", callback_data="model:deepcoder")],
        [InlineKeyboardButton("mistral-small", callback_data="model:mistral-small")],
        [InlineKeyboardButton("deepseek-r1", callback_data="model:deepseek-r1")],
        [InlineKeyboardButton("deepseek-chat", callback_data="model:deepseek-chat")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите модель:", reply_markup=reply_markup)

async def callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик выбора модели."""
    query = update.callback_query
    user_id = query.from_user.id
    model = query.data.split(":")[1]
    context.user_data["model"] = model
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
        "deepseek-r1": "deepseek/deepseek-r1:free",
        "deepseek-chat": "deepseek/deepseek-chat:free"
    }.get(model, "agentica-org/deepcoder-14b-preview:free")

    log_event("New request", user_id, f"Model: {model_id}\nMessage: '{message_text}'")

    try:
        # Запрос к OpenRouter API
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("API key for OpenRouter is not configured")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": message_text}]
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0
            )
            response.raise_for_status()
            response_data = response.json()
            reply_text = response_data["choices"][0]["message"]["content"]
            log_event("API success", user_id, f"Response length: {len(reply_text)} chars")

        # Проверяем, содержит ли ответ код (ищем ``` или упоминание кода)
        if "```" in reply_text or "код" in message_text.lower():
            # Разделяем ответ на код и пояснение
            code_match = re.search(r"```(?:python)?\n([\s\S]*?)\n```", reply_text)
            if code_match:
                code = code_match.group(1)
                # Всё, что вне блока кода, считаем пояснением
                explanation = re.sub(r"```(?:python)?\n[\s\S]*?\n```", "", reply_text).strip()
            else:
                # Если блок кода не найден, считаем весь текст кодом
                code = reply_text
                explanation = ""
            
            # Форматируем сообщение с копируемым кодом
            formatted_message = format_code_message(code, explanation)
        else:
            # Для некодовых ответов экранируем весь текст
            formatted_message = escape_markdown_v2(reply_text)

        # Отправляем сообщение, разбивая на части, если нужно
        try:
            for part in split_message(formatted_message):
                await update.message.reply_text(
                    part,
                    parse_mode=telegram.constants.ParseMode.MARKDOWN_V2
                )
        except telegram.error.BadRequest as e:
            log_event("Markdown error", user_id, f"Falling back to plain text: {str(e)}")
            # Откат на обычный текст
            for part in split_message(reply_text):
                await update.message.reply_text(part)

    except httpx.HTTPStatusError as e:
        error_msg = f"API error ❌ Ошибка {e.response.status_code}: {e.response.text}"
        log_event("API error", user_id, error_msg)
        await update.message.reply_text(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        log_event("Error", user_id, error_msg, exc_info=True)
        await update.message.reply_text("Произошла ошибка, попробуйте позже.")

def handle_shutdown(app: Application):
    """Обработчик сигналов для graceful shutdown."""
    def shutdown(signum, frame):
        logger.info("Received shutdown signal, stopping bot")
        app.stop()
    return shutdown

def main():
    """Запуск бота."""
    log_event("BOT STARTED", 0, "=== BOT STARTED ===")
    log_event("Available models", 0, "Available models: ['mistral-small', 'deepseek-r1', 'deepseek-chat', 'deepcoder']")
    
    app = Application.builder().token("7888772385:AAEGpPDVxsFBqjskcI__taTaa01VveBrVPM").build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(callback_query, pattern="model:.*"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Настройка graceful shutdown
    shutdown_handler = handle_shutdown(app)
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    log_event("Polling started", 0, "Polling started")
    app.run_polling()

if __name__ == "__main__":
    main()
