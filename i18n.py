"""
Модуль локализации (i18n) и быстрой отрисовки текста с кэшем для «Сиқырлы айна».
Поддерживает все казахские буквы: Ә Ғ Қ Ң Ө Ұ Ү Һ І.
По умолчанию всегда казахский язык ("kk"), русский ("ru") как запасной по клавише L.
"""
import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import config

_FALLBACK_T = {
    "kk": dict(
        title="Сиқырлы айна", choose="Кейіпкеріңді таңда!", score="Ұпай", time="Уақыт", sec="сек",
        well_done="Жарайсың!", awesome="Керемет!", victory="Жеңіс!", ready_q="Дайынсың ба?",
        lets_start="Бастаймыз!", raise_hand="Қолыңды жоғары көтер!", raise_both_hands="Екі қолыңды көтер",
        play_again="Қайта ойнау", next_level="Келесі деңгей", stand_in_frame="Кадрға тұр",
        fly="Ұшу", beam="Сәуле", shield="Қалқан", monster="Құбыжық", hero_word="Қаһарман", batyr="Батыр",
        fly_bang="ҰШУ!", shield_bang="ҚАЛҚАН!", beam_bang="СӘУЛЕ!",
        pause="Кадрға тұр!", medal="ҚАҺАРМАН МЕДАЛІ!", medal_praise="Жарайсың, батыр!",
        again="Қайта ойнау", hint="Екі қолды көтер — Ұшу  |  Қол жанға — Сәуле  |  Қол айқас — Қалқан",
        stand="Кадрға тұр", hold="Қолыңды карточка үстінде ұста",
        tut_title="Жаттығу: Қимылды қайтала!",
        tut_fly="Екі қолыңды көтер!", tut_beam="Қолыңды жоғары көтер!", tut_shield="Қолыңды айқастыр!",
        tut_hold_steady="Жарайсың!", tut_try_bigger="Суреттегідей қайтала!",
        tut_skip_hint="Тәрбиеші: S басыңыз немесе батырманы ұстаңыз",
        btn_start="Бастаймыз!", btn_next="Келесі деңгей", btn_again="Қайта ойнау", btn_skip="Өткізу",
        mission_stars="5 жұлдыз жина!", mission_monsters="Құбыжықты жең!", mission_boss="Бас құбыжықты жең!",
        combo="Комбо", energy="Күш", level="Деңгей", low_light="Жарық аз! Жарықты қосыңыз",
    ),
    "ru": dict(
        title="Волшебное зеркало", choose="Выбери своего героя!", score="Очки", time="Время", sec="сек",
        well_done="Молодец!", awesome="Супер!", victory="Победа!", ready_q="Готов?",
        lets_start="Начинаем!", raise_hand="Подними руку вверх!", raise_both_hands="Подними обе руки",
        play_again="Играть снова", next_level="Следующий уровень", stand_in_frame="Встань в кадр",
        fly="Полёт", beam="Луч", shield="Щит", monster="Монстр", hero_word="Герой", batyr="Батыр",
        fly_bang="ПОЛЁТ!", shield_bang="ЩИТ!", beam_bang="ЛУЧ!",
        pause="Встань в кадр!", medal="МЕДАЛЬ ГЕРОЯ!", medal_praise="Молодец, батыр!",
        again="Играть снова", hint="Обе руки вверх — Полёт  |  Рука в сторону — Луч  |  Скрести руки — Щит",
        stand="Встань в кадр", hold="Держи руку над карточкой",
        tut_title="Обучение: Повтори движение!",
        tut_fly="Подними обе руки!", tut_beam="Вытяни руку в сторону!", tut_shield="Скрести руки на груди!",
        tut_hold_steady="Молодец!", tut_try_bigger="Повтори за силуэтом!",
        tut_skip_hint="Воспитатель: нажмите S или кнопку пропуска",
        btn_start="Начинаем!", btn_next="Следующий уровень", btn_again="Играть снова", btn_skip="Пропустить",
        mission_stars="Собери 5 звёзд!", mission_monsters="Победи монстра!", mission_boss="Победи босса!",
        combo="Комбо", energy="Энергия", level="Уровень", low_light="Мало света! Включите освещение",
    ),
}


def load_translations():
    res = {"kk": dict(_FALLBACK_T["kk"]), "ru": dict(_FALLBACK_T["ru"])}
    for code in ("kk", "ru"):
        p = os.path.join(config.BASE_DIR, "lang", f"{code}.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if not k.startswith("_") and isinstance(v, str):
                        res[code][k] = v
            except Exception as e:
                print(f"[LANG] Ошибка чтения {p}: {e}")
    return res


T = load_translations()


def get_lang():
    return config.LANG


def set_lang(code):
    config.LANG = "ru" if code == "ru" else "kk"
    return config.LANG


def toggle_lang():
    config.LANG = "ru" if config.LANG == "kk" else "kk"
    return config.LANG


def tr(key, default=""):
    lang = config.LANG
    return T.get(lang, T["kk"]).get(key, _FALLBACK_T.get(lang, _FALLBACK_T["kk"]).get(key, default or key))


FONT_PATHS = [
    "C:/Windows/Fonts/NotoSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_fonts = {}
_txt_cache = {}
KAZAKH_CHARS = "ӘәҒғҚқҢңӨөҰұҮүҺһІі"


def font(sz):
    if sz not in _fonts:
        for pth in FONT_PATHS:
            if os.path.exists(pth):
                try:
                    _fonts[sz] = ImageFont.truetype(pth, sz)
                    break
                except Exception:
                    pass
        else:
            _fonts[sz] = ImageFont.load_default()
    return _fonts[sz]


def verify_kazakh_glyphs():
    """Проверка отрисовки всех специфических казахских букв Ә Ғ Қ Ң Ө Ұ Ү Һ І при старте."""
    f = font(36)
    test_img = Image.new("RGBA", (600, 60), (0, 0, 0, 0))
    d = ImageDraw.Draw(test_img)
    d.text((10, 5), KAZAKH_CHARS, font=f, fill=(255, 255, 255, 255))
    arr = np.asarray(test_img)
    return int(np.count_nonzero(arr[:, :, 3])) > 100


def _get_txt_sprite(s, size, col):
    """Кэшируем отрендеренный PIL-текст с обводкой (BGR + инверт-альфа uint8) для максимального FPS."""
    key = (s, size, col)
    if key in _txt_cache:
        return _txt_cache[key]
    if len(_txt_cache) > 512:
        _txt_cache.clear()
    f = font(size)
    sw = max(2, size // 14)
    l, t, r, b = f.getbbox(s, stroke_width=sw)
    pad = sw + 4
    w = max(1, (r - l) + pad * 2)
    h = max(1, (b - t) + pad * 2)
    rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(rgba)
    draw.text((pad - l, pad - t), s, font=f, fill=(col[2], col[1], col[0], 255),
              stroke_width=sw, stroke_fill=(10, 14, 24, 240))
    arr = np.asarray(rgba)
    a3 = cv2.merge([arr[:, :, 3], arr[:, :, 3], arr[:, :, 3]])
    fg_premult = cv2.multiply(arr[:, :, :3], a3, scale=1.0 / 255.0)
    inv_a3 = cv2.bitwise_not(a3)
    _txt_cache[key] = (fg_premult, inv_a3, w, h, pad - l, pad - t)
    return _txt_cache[key]


def txt(img, s, x, y, size=40, col=(255, 255, 255), center=False):
    if not s:
        return
    fg_pre, inv_a3, w, h, ox, oy = _get_txt_sprite(str(s), int(size), tuple(int(c) for c in col))
    x0 = int(x - w // 2) if center else int(x - ox)
    y0 = int(y - oy // 3)
    hh, ww = img.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(ww, x0 + w), min(hh, y0 + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
    roi = img[dy0:dy1, dx0:dx1]
    img[dy0:dy1, dx0:dx1] = cv2.add(cv2.multiply(roi, inv_a3[sy0:sy1, sx0:sx1], scale=1.0 / 255.0),
                                     fg_pre[sy0:sy1, sx0:sx1])
