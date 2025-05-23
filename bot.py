import os
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Конфигурация (ЗАМЕНИТЕ НА СВОИ ДАННЫЕ!)
OPENROUTER_API_KEY = "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"
BOT_TOKEN = "7888772385:AAEGpPDVxsFBqjskcI__taTaa01VveBrVPM"
MODEL = "mistralai/devstral-small:free"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Я бот с искусственным интеллектом. Задайте мне вопрос!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Формируем запрос к OpenRouter
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY.strip()}",
            "HTTP-Referer": "https://github.com",  # Обязательный заголовок
            "X-Title": "Telegram AI Bot"          # Идентификатор приложения
        }
        
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": update.message.text}]
        }

        # Отправка запроса
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            # Обработка ответа
            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
                await update.message.reply_text(answer)
            else:
                error_msg = f"❌ Ошибка {response.status_code}: {response.text}"
                await update.message.reply_text(error_msg)

    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка: {str(e)}")

if __name__ == "__main__":
    # Проверка обязательных заголовков
    if not OPENROUTER_API_KEY.startswith("sk-or-"):
        print("ОШИБКА: Неверный формат API ключа OpenRouter!")
        exit(1)

    print("🟢 Запуск бота...")
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    app.run_polling()
