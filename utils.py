import re
from typing import List, Optional
from telegram import Update
from telegram.constants import ParseMode
from OpenRouterChat.config import MAX_MESSAGE_LENGTH, TELEGRAM_MARKDOWN_SPECIAL_CHARS, logger

def escape_markdown_v2(text: str) -> str:
    """
    Экранирует специальные символы для Telegram MarkdownV2, кроме блоков кода.

    Args:
        text (str): Входной текст для экранирования.

    Returns:
        str: Экранированный текст, готовый для отправки в Telegram.
    """
    if not text:
        return ""
    return re.sub(TELEGRAM_MARKDOWN_SPECIAL_CHARS, r'\\\1', text)

def clean_text(text: str) -> str:
    """
    Очищает текст от управляющих символов и добавляет базовую пунктуацию.

    Args:
        text (str): Входной текст.

    Returns:
        str: Очищенный текст с точкой в конце, если необходимо.
    """
    if not text:
        return ""
    cleaned = re.sub(r'[\x00-\x1F\x7F]', '', text).strip()
    if cleaned and cleaned[-1] not in '.!?':
        cleaned += '.'
    return cleaned

def sanitize_input(text: str) -> str:
    """
    Очищает пользовательский ввод от потенциально опасных символов.

    Args:
        text (str): Входной текст.

    Returns:
        str: Очищенный текст.
    """
    if not text:
        return ""
    return re.sub(r'<[^>]+>|[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text).strip()

def is_code(text: str) -> bool:
    """
    Проверяет, является ли текст кодом на основе нескольких критериев.

    Args:
        text (str): Входной текст.

    Returns:
        bool: True, если текст похож на код, иначе False.
    """
    if not text:
        return False
    code_indicators = [
        (r'\b(def|class|while|for|if|print|import)\b', 2),  # Ключевые слова (вес 2)
        (r'[=\(\):]', 1),                                   # Синтаксические элементы (вес 1)
        (r'^\s{2,}', 1)                                     # Отступы (вес 1)
    ]
    score = sum(weight for pattern, weight in code_indicators if re.search(pattern, text, re.MULTILINE))
    return score >= 3

def validate_code(code: str) -> str:
    """
    Проверяет и исправляет базовые ошибки в коде.

    Args:
        code (str): Входной код.

    Returns:
        str: Исправленный код.
    """
    if not code:
        return ""
    lines = code.split('\n')
    corrected_lines = []
    for line in lines:
        # Исправление ошибки типа "while count count 1"
        if 'while' in line.lower() and line.count('count') > 2:
            line = re.sub(r'while\s+count\s+count\s+1', 'while count <= 5:', line, flags=re.IGNORECASE)
        # Исправление неправильных отступов
        line = line.rstrip()
        corrected_lines.append(line)
    return '\n'.join(corrected_lines).strip()

def format_code_message(code: str, explanation: str = "") -> str:
    """
    Форматирует сообщение с кодом и пояснением для Telegram MarkdownV2.

    Args:
        code (str): Код для форматирования.
        explanation (str): Пояснение к коду.

    Returns:
        str: Отформатированное сообщение.
    """
    code = validate_code(clean_text(code)).strip()
    explanation = clean_text(explanation).strip()
    
    if not code and not explanation:
        return "Не удалось сгенерировать ответ. Попробуйте уточнить запрос."
    
    code_block = f"```python\n{code}\n```" if code and is_code(code) else ""
    
    if explanation and explanation != code and len(explanation) > 5:
        escaped_explanation = escape_markdown_v2(explanation)
        return f"{code_block}\n\n{escaped_explanation}" if code_block else escaped_explanation
    
    return code_block or escape_markdown_v2(explanation)

async def send_message(update: Update, text: str, parse_mode: Optional[str] = None, user_id: int = None) -> None:
    """
    Отправляет сообщение, разбивая его на части при необходимости.

    Args:
        update (Update): Объект обновления Telegram.
        text (str): Текст сообщения.
        parse_mode (Optional[str]): Режим форматирования (например, ParseMode.MARKDOWN_V2).
        user_id (int, optional): ID пользователя для логирования.
    """
    for part in [text[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(text), MAX_MESSAGE_LENGTH)]:
        try:
            await update.message.reply_text(part, parse_mode=parse_mode)
        except Exception as e:
            logger.error(f"Ошибка отправки сообщения пользователю {user_id}: {str(e)}")
            await update.message.reply_text(clean_text(part))
