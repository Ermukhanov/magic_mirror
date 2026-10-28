"""
Модуль конфигурации «Сиқырлы айна» (Magic Mirror v2).
Загружает настройки из config.json с безопасными значениями по умолчанию.
"""
import os
import json
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
LOG_PATH = os.path.join(BASE_DIR, "magic_mirror.log")
STATS_PATH = os.path.join(BASE_DIR, "stats.json")

_DEFAULTS = {
    "width": 1280,
    "height": 720,
    "pose_width": 384,
    "pose_height": 216,
    "pose_complexity": 0,
    "game_time_sec": 60,
    "hold_select_sec": 1.0,
    "hold_action_sec": 0.20,
    "hold_tutorial_sec": 0.9,
    "hold_button_sec": 1.0,
    "default_lang": "kk",
    "show_fps": True,
    "fullscreen": True,
    "volume": 0.8,
    "ai_transform": "auto",
    "video_worlds": True,
    "fx_clips": True,
    "hero_art": True,
    "game_layer": True,
    "rich_worlds": True,
    "sketch_conditioning": True,
    "hands_live": True,
    "procedural_armor": True,
}


def load_config():
    cfg = dict(_DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    # Облегчённый профиль для слабых ноутбуков (запуск через start_lite.bat: MM_PROFILE=lite)
    if os.environ.get("MM_PROFILE", "").lower() == "lite":
        try:
            with open(os.path.join(BASE_DIR, "config_lite.json"), "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg


CFG = load_config()

W = int(CFG["width"])
H = int(CFG["height"])
SW = int(CFG["pose_width"])
SH = int(CFG["pose_height"])
POSE_COMPLEXITY = int(CFG["pose_complexity"])
GAME_TIME = int(CFG["game_time_sec"])
HOLD_SELECT = float(CFG["hold_select_sec"])
HOLD_ACTION = float(CFG["hold_action_sec"])
HOLD_TUTORIAL = float(CFG["hold_tutorial_sec"])
HOLD_BUTTON = float(CFG["hold_button_sec"])
LANG = str(CFG.get("default_lang", "kk"))
SHOW_FPS = bool(CFG.get("show_fps", True))
FULLSCREEN = bool(CFG.get("fullscreen", True))
DEFAULT_VOLUME = float(CFG.get("volume", 0.8))

# Флаги модулей (включаются и выключаются в config.json и в панели воспитателя [T])
AI_TRANSFORM = str(CFG.get("ai_transform", CFG.get("ai_mode", "auto")))
VIDEO_WORLDS = bool(CFG.get("video_worlds", True))
FX_CLIPS = bool(CFG.get("fx_clips", True))
HERO_ART = bool(CFG.get("hero_art", True))
GAME_LAYER = bool(CFG.get("game_layer", True))
RICH_WORLDS = VIDEO_WORLDS
SKETCH_CONDITIONING = bool(CFG.get("sketch_conditioning", True))
HANDS_LIVE = bool(CFG.get("hands_live", True))
PROCEDURAL_ARMOR = bool(CFG.get("procedural_armor", False))

AA = cv2.LINE_AA


def save_config_flags():
    """Сохраняет текущие состояния флагов модулей обратно в config.json."""
    try:
        data = load_config()
        data["ai_transform"] = str(AI_TRANSFORM)
        data["video_worlds"] = bool(VIDEO_WORLDS)
        data["fx_clips"] = bool(FX_CLIPS)
        data["hero_art"] = bool(HERO_ART)
        data["game_layer"] = bool(GAME_LAYER)
        data["rich_worlds"] = bool(VIDEO_WORLDS)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
