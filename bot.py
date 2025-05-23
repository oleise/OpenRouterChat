import httpx
import logging
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log")
    ]
)
logger = logging.getLogger(__name__)

# Конфигурация (ваши данные)
OPENROUTER_API_KEY = "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"
BOT_TOKEN = "7888772385:AAEGpPDVxsFBqjskcI__taTaa01VveBrVPM"

# Доступные модели (ваш список)
MODELS = {
    "mistral-small": "mistralai/devstral-small:free",
    "deepseek-r1": "deepseek/deepseek-r1:free",
    "deepseek-chat": "deepseek/deepseek-chat-v3-0324:free",
    "deepcoder": "agentica-org/deepcoder-14b-preview:free"
}

current_model = MODELS["mistral-small"]

def format_code_block(text: str) -> str:
    """Форматирует код для Telegram с определением языка"""
    if "```" in text:
        return text  # Уже отформатирован
    return f"```python\n{text}\n```" if any(kw in text.lower() for kw in ["def ", "import ", "class ", "print("]) else f"```\n{text}\n```"

def log_event(event: str, user_id: int = None, details: str = ""):
    """Логирование событий с timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_info = f"User {user_id}: " if user_id else ""
    log_message = f"[{timestamp}] {user_info}{event} {details}"
    logger.info(log_message)
    print(log_message)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    log_event("Start command", user.id)
    
    keyboard = [
        [InlineKeyboardButton("Mistral Small", callback_data="mistral-small")],
        [InlineKeyboardButton("DeepSeek R1", callback_data="deepseek-r1")],
        [InlineKeyboardButton("DeepSeek Chat", callback_data="deepseek-chat")],
        [InlineKeyboardButton("DeepCoder", callback_data="deepcoder")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🤖 Выберите модель ИИ:\n"
        f"Текущая: {current_model.split('/')[-1].replace(':free', '')}",
        reply_markup=reply_markup,
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_model
    query = update.callback_query
    user = query.from_user
    model_key = query.data
    
    await query.answer()
    current_model = MODELS[model_key]
    
    log_event(f"Model changed", user.id, f"New model: {model_key}")
    
    await query.edit_message_text(
        f"✅ Выбрана модель: {current_model.split('/')[-1].replace(':free', '')}\n"
        "Отправьте мне сообщение для обработки."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_message = update.message.text
    
    log_event("New request", user.id, f"Model: {current_model}\nMessage: '{user_message[:50]}...'")
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY.strip()}",
            "HTTP-Referer": "https://github.com",
            "X-Title": "Telegram AI Bot",
        }
        
        payload = {
            "model": current_model,
            "messages": [{"role": "user", "content": user_message}],
        }

        log_event("API request", user.id, f"Model: {current_model}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            
            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
                
                # Форматируем код в ответе
                formatted_answer = format_code_block(answer)
                
                log_event("API success", user.id, f"Response length: {len(answer)} chars")
                
                # Отправляем с parse_mode='MarkdownV2' для форматирования
                await update.message.reply_text(
                    formatted_answer,
                    parse_mode='MarkdownV2'
                )
            else:
                error_msg = f"❌ Ошибка {response.status_code}: {response.text[:200]}"
                log_event("API error", user.id, error_msg)
                await update.message.reply_text(error_msg)

    except httpx.TimeoutException:
        error_msg = "⌛ Таймаут запроса (30 сек)"
        log_event("Timeout", user.id, error_msg)
        await update.message.reply_text(error_msg)
        
    except Exception as e:
        error_msg = f"⚠️ Ошибка: {str(e)}"
        log_event("Error", user.id, error_msg, exc_info=True)
        await update.message.reply_text(error_msg)

async def model_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)

if __name__ == "__main__":
    log_event("=== BOT STARTED ===")
    log_event(f"Available models: {list(MODELS.keys())}")
    
    try:
        app = Application.builder().token(BOT_TOKEN).build()
        
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("model", model_list))
        app.add_handler(CallbackQueryHandler(button_handler))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        log_event("Polling started")
        app.run_polling()
        
    except Exception as e:
        log_event("FATAL ERROR", details=str(e), exc_info=True)
        raise
