import logging
from typing import Dict
from environs import Env
from logging.handlers import RotatingFileHandler

# Инициализация переменных окружения
env = Env()
env.read_env()

# Переменные окружения
# TELEGRAM_TOKEN: задается в переменной окружения TELEGRAM_TOKEN (получить у @BotFather)
# OPENROUTER_API_KEY: задается в переменной окружения OPENROUTER_API_KEY (получить на openrouter.ai)
TELEGRAM_TOKEN: str = env.str("TELEGRAM_TOKEN", default=None)
OPENROUTER_API_KEY: str = env.str("OPENROUTER_API_KEY", default=None)
API_TIMEOUT: float = env.float("API_TIMEOUT", default=10.0)
LOG_FILE: str = env.str("LOG_FILE", default="bot.log")
LOG_MAX_BYTES: int = env.int("LOG_MAX_BYTES", default=10_000_000)
LOG_BACKUP_COUNT: int = env.int("LOG_BACKUP_COUNT", default=5)

# Проверка обязательных переменных
if not all([TELEGRAM_TOKEN, OPENROUTER_API_KEY]):
    logging.error("Отсутствуют обязательные переменные окружения: TELEGRAM_TOKEN, OPENROUTER_API_KEY")
    raise ValueError("Переменные TELEGRAM_TOKEN и OPENROUTER_API_KEY должны быть заданы.")

# Константы
MAX_MESSAGE_LENGTH: int = 4096
TELEGRAM_MARKDOWN_SPECIAL_CHARS: str = r'([_\*\[\]\(\)~`>\#\+\-\=\|\{\}\.\!])'

# Маппинг моделей
MODELS: Dict[str, Dict[str, str]] = {
    "deepcoder": {"id": "agentica-org/deepcoder-14b-preview:free", "name": "DeepCoder"},
    "mistral-small": {"id": "mistralai/devstral-small:free", "name": "Mistral Small"},
    "deepseek-r1": {"id": "deepseek/deepseek-r1:free", "name": "DeepSeek R1"},
    "deepseek-chat": {"id": "deepseek/deepseek-chat:free", "name": "DeepSeek Chat"}
}

# Настройка логирования с ротацией
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT),
        logging.StreamHandler()
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
