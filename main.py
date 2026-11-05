import os
import asyncio
import logging

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart, Command
from google import genai
from google.genai import types as genai_types
import pypdf
import docx
from dotenv import load_dotenv

load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aichatbot")

# ---------- Настройки ----------
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_TOKEN")

# Основная модель (можно переопределить в .env: MODEL_NAME=...)
PRIMARY_MODEL = os.getenv("MODEL_NAME", "gemini-flash-latest")
# Запасные модели: если основная перегружена или не отвечает, пробуем следующую
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
MODELS = list(dict.fromkeys([PRIMARY_MODEL, *FALLBACK_MODELS]))  

REQUEST_TIMEOUT = 25              
MAX_HISTORY = 20                 
MAX_FILE_SIZE = 20 * 1024 * 1024   
MAX_DOC_CHARS = 30000             
TG_MESSAGE_LIMIT = 4000            
SUPPORTED_EXTS = {".txt", ".pdf", ".docx"}

CHAT_SYSTEM_PROMPT = "Ты полезный, вежливый и эрудированный ассистент."
DOC_SYSTEM_PROMPT = (
    "Ты ассистент по анализу документов. "
    "Анализируй текст и отвечай на запрос пользователя четко и структурировано."
)

if not BOT_TOKEN or not GEMINI_API_KEY:
    raise RuntimeError("Проверьте BOT_TOKEN и GEMINI_TOKEN в файле .env")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
ai_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options={"timeout": REQUEST_TIMEOUT * 1000}, 
)

histories: dict[int, list] = {}


# ---------- Работа с Gemini ----------
def make_content(role: str, text: str) -> genai_types.Content:
    return genai_types.Content(role=role, parts=[genai_types.Part(text=text)])


async def ask_gemini(contents: list, system_instruction: str) -> str:
    last_error: Exception | None = None

    for model in MODELS:
        try:
            response = await asyncio.wait_for(
                ai_client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_instruction
                    ),
                ),
                timeout=REQUEST_TIMEOUT + 5,
            )
            log.info("Ответ получен от модели %s", model)
            return response.text or "Не удалось получить ответ."
        except Exception as e:
            log.warning("Модель %s не ответила: %s: %s", model, type(e).__name__, e)
            last_error = e

    raise last_error or RuntimeError("Нет доступных моделей")


def extract_text_from_file(file_path: str, ext: str) -> str:
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == ".pdf":
        reader = pypdf.PdfReader(file_path)
        parts = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                parts.append(extracted)
        return "\n".join(parts)

    if ext == ".docx":
        document = docx.Document(file_path)
        return "\n".join(p.text for p in document.paragraphs if p.text)

    return ""


async def send_long(message: types.Message, text: str) -> None:
    text = text or "Не удалось получить ответ."
    for i in range(0, len(text), TG_MESSAGE_LIMIT):
        await message.answer(text[i:i + TG_MESSAGE_LIMIT])


# ---------- Команды ----------
@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я ваш ИИ-помощник на базе Google Gemini.\n"
        "Вы можете задать мне любой вопрос или отправить документ (PDF, TXT, DOCX), "
        "чтобы я проанализировал его содержимое!\n\n"
        "Команда /reset очищает историю диалога."
    )


@dp.message(Command("reset"))
async def cmd_reset(message: types.Message):
    histories.pop(message.chat.id, None)
    await message.answer("История диалога очищена.")


@dp.message(F.text)
async def handle_text(message: types.Message):
    await bot.send_chat_action(message.chat.id, "typing")

    history = histories.setdefault(message.chat.id, [])
    history.append(make_content("user", message.text))

    try:
        answer = await ask_gemini(history, CHAT_SYSTEM_PROMPT)
        history.append(make_content("model", answer))
        if len(history) > MAX_HISTORY:
            del history[:-MAX_HISTORY]
        await send_long(message, answer)
    except Exception:
        history.pop()
        log.exception("Ошибка при обращении к Gemini (текст)")
        await message.answer("ИИ сейчас недоступен или перегружен. Попробуйте ещё раз через минуту.")


# ---------- Документы ----------
@dp.message(F.document)
async def handle_document(message: types.Message):
    document = message.document
    filename = document.file_name or "file"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in SUPPORTED_EXTS:
        await message.answer("Пожалуйста, отправьте файл в формате PDF, TXT или DOCX.")
        return

    if document.file_size and document.file_size > MAX_FILE_SIZE:
        await message.answer("Файл слишком большой. Максимальный размер — 20 МБ.")
        return

    await message.answer("Скачиваю и читаю файл...")

    os.makedirs("downloads", exist_ok=True)
    local_path = os.path.join("downloads", f"{document.file_unique_id}{ext}")

    try:
        file_info = await bot.get_file(document.file_id)
        await bot.download_file(file_info.file_path, local_path)

        extracted_text = await asyncio.to_thread(extract_text_from_file, local_path, ext)

        if not extracted_text.strip():
            await message.answer(
                "Не удалось извлечь текст из файла "
                "(возможно, это скан без распознанного текста)."
            )
            return

        truncated = len(extracted_text) > MAX_DOC_CHARS
        text_for_model = extracted_text[:MAX_DOC_CHARS]

        user_request = message.caption or "Сделай краткую выжимку этого файла и опиши ключевые мысли."
        full_prompt = f"Запрос пользователя: {user_request}\n\nТекст документа:\n{text_for_model}"

        await bot.send_chat_action(message.chat.id, "typing")

        answer = await ask_gemini([make_content("user", full_prompt)], DOC_SYSTEM_PROMPT)

        if truncated:
            answer += (
                f"\n\n⚠️ Документ длинный, проанализированы только первые {MAX_DOC_CHARS} символов."
            )
        await send_long(message, answer)

    except Exception:
        log.exception("Ошибка при обработке файла")
        await message.answer("Ошибка при обработке файла. Попробуйте другой файл или повторите позже.")
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


async def main():
    log.info("Бот запущен. Модели по порядку: %s", ", ".join(MODELS))
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Бот остановлен.")