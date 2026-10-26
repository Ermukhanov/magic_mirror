"""
Генератор офлайн-озвучки для игры «Сиқырлы айна» через edge-tts.
Использует голоса kk-KZ-AigulNeural и kk-KZ-DauletNeural для казахского языка
и ru-RU-SvetlanaNeural / ru-RU-DmitryNeural для запасного русского языка.

Запуск:
    python tools/gen_voice.py
Файлы сохраняются в:
    sounds/voice/kk/*.mp3
    sounds/voice/ru/*.mp3
"""
import os
import json
import asyncio
import edge_tts

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANG_DIR = os.path.join(BASE_DIR, "lang")
VOICE_DIR = os.path.join(BASE_DIR, "sounds", "voice")

# Ключевые фразы для озвучки в игре (обучение, выбор героев, миссии, похвала, финал)
VOICE_KEYS = [
    "hello",
    "hello_stand",
    "come_closer",
    "hold_hero",
    "calib_title",
    "calib_done",
    "choose",
    "well_done",
    "awesome",
    "victory",
    "ready_q",
    "lets_start",
    "raise_hand",
    "raise_both_hands",
    "play_again",
    "next_level",
    "stand_in_frame",
    "fly",
    "beam",
    "shield",
    "medal_praise",
    "tut_fly",
    "tut_beam",
    "tut_shield",
    "tut_try_bigger",
    "mission_stars",
    "mission_monsters",
    "mission_boss",
    "hero_alpamys",
    "hero_tomiris",
    "hero_spider",
    "hero_elsa",
    "hero_ladybug",
    "hero_iron",
    "hero_astronaut",
    "hero_fairy",
    "spec_alpamys",
    "spec_tomiris",
    "spec_spider",
    "spec_elsa",
    "spec_ladybug",
    "spec_iron",
    "spec_astronaut",
    "spec_fairy",
]

# Чередуем/подбираем голоса: женский добрый голос Aigul для подсказок и феи/принцесс,
# богатырский мужской голос Daulet для батыров и суперударов
DAULET_KEYS = {
    "hero_alpamys", "hero_spider", "hero_iron", "hero_astronaut",
    "spec_alpamys", "spec_spider", "spec_iron", "spec_astronaut",
    "mission_boss", "medal_praise", "victory",
}


async def gen_one(text, voice, out_path):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
        return True
    for attempt_voice in (voice, "ru-RU-SvetlanaNeural" if voice.startswith("ru-") else "kk-KZ-AigulNeural"):
        try:
            comm = edge_tts.Communicate(text=text, voice=attempt_voice, rate="+5%")
            await comm.save(out_path)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
                return True
        except Exception as e:
            await asyncio.sleep(0.4)
    print(f"[WARN] Не удалось сгенерировать {out_path} ({voice})")
    return False


async def main():
    for lang_code in ("kk", "ru"):
        json_path = os.path.join(LANG_DIR, f"{lang_code}.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        out_dir = os.path.join(VOICE_DIR, lang_code)
        os.makedirs(out_dir, exist_ok=True)

        tasks = []
        for k in VOICE_KEYS:
            txt_val = data.get(k)
            if not txt_val or not isinstance(txt_val, str):
                continue
            # Убираем лишние символы тире для более естественного произношения
            clean_txt = txt_val.replace(" — ", ", ")
            if lang_code == "kk":
                voice = "kk-KZ-DauletNeural" if k in DAULET_KEYS else "kk-KZ-AigulNeural"
            else:
                voice = "ru-RU-DmitryNeural" if k in DAULET_KEYS else "ru-RU-SvetlanaNeural"
            out_file = os.path.join(out_dir, f"{k}.mp3")
            tasks.append((k, clean_txt, voice, out_file))

        print(f"[{lang_code.upper()}] Генерация {len(tasks)} голосовых фраз в {out_dir}...")
        sem = asyncio.Semaphore(4)

        async def worker(item):
            k, clean_txt, voice, out_file = item
            async with sem:
                ok = await gen_one(clean_txt, voice, out_file)
                if ok:
                    print(f"  OK: {lang_code}/{k}.mp3 ({voice}) -> {clean_txt}")

        await asyncio.gather(*(worker(it) for it in tasks))


if __name__ == "__main__":
    asyncio.run(main())
