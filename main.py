import signal
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from .config import TELEGRAM_TOKEN, logger
from .handlers import start, callback_query, handle_message

def handle_shutdown(app: Application) -> callable:
    """
    Создает обработчик сигналов для graceful shutdown.

    Args:
        app (Application): Объект приложения Telegram.

    Returns:
        callable: Функция обработки сигналов.
    """
    def shutdown(signum, frame):
        logger.info("Получен сигнал завершения, остановка бота")
        app.stop()
    return shutdown

def main() -> None:
    """Запуск бота."""
    log_event = logger.info
    log_event("BOT STARTED", extra={"user_id": 0, "event_type": "Старт бота"})
    log_event(f"Доступные модели: {list(MODELS.keys())}", extra={"user_id": 0, "event_type": "Конфигурация"})

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(callback_query, pattern="model:.*"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Настройка graceful shutdown
    shutdown_handler = handle_shutdown(app)
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    log_event("Начало опроса", extra={"user_id": 0, "event_type": "Опрос"})
    app.run_polling(poll_interval=0.1)

if __name__ == "__main__":
    from .config import MODELS, logger
    main()
