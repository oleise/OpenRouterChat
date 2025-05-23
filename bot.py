import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

# Конфигурация
OPENROUTER_API_KEY = "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"  # Замените на свой ключ
BOT_TOKEN = "7888772385:AAEGpPDVxsFBqjskcI__taTaa01VveBrVPM"  # От @BotFather

# Доступные модели
MODELS = {
    "mistral-small": "mistralai/devstral-small:free",
    "deepseek": "deepseek/deepseek-r1:free",
    "deepseekv3": "deepseek/deepseek-chat-v3-0324:free",
    "deepcoder": "agentica-org/deepcoder-14b-preview:free",
}

# Глобальная переменная для хранения выбранной модели (по умолчанию)
current_model = MODELS["mistral-small"]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Mistral Small", callback_data="mistral-small")],
        [InlineKeyboardButton("DeepSeek R1", callback_data="deepseek")],
        [InlineKeyboardButton("DeepSeek V3", callback_data="deepseekv3")],
        [InlineKeyboardButton("DeepCoder", callback_data="deepcoder")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🤖 Выберите модель ИИ:\n"
        f"Сейчас выбрана: {current_model.split('/')[-1]}",
        reply_markup=reply_markup,
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_model
    query = update.callback_query
    await query.answer()
    
    model_key = query.data
    current_model = MODELS[model_key]
    
    await query.edit_message_text(
        f"✅ Выбрана модель: {current_model.split('/')[-1]}\n"
        "Отправьте мне сообщение для обработки."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY.strip()}",
            "HTTP-Referer": "https://github.com",
            "X-Title": "Telegram AI Bot",
        }
        
        payload = {
            "model": current_model,
            "messages": [{"role": "user", "content": update.message.text}],
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            
            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
                await update.message.reply_text(answer)
            else:
                error_msg = f"❌ Ошибка {response.status_code}: {response.text}"
                await update.message.reply_text(error_msg)

    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка: {str(e)}")

async def model_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)

if __name__ == "__main__":
    print("🟢 Запуск бота...")
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("model", model_list))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    app.run_polling()
