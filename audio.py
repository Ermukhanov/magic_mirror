"""
Модуль звука и офлайн-озвучки «Сиқырлы айна» на базе pygame.mixer.
- Проигрывает готовые голосовые фразы из sounds/voice/kk/*.mp3 и sounds/voice/ru/*.mp3
- Проигрывает фоновую музыку героев (из sounds/music/*.mp3 или процедурно синтезированный кюй/мелодию)
- Проигрывает звуковые эффекты действий из sounds/sfx/*.wav или синтезирует их через numpy
"""
import os
import time
import numpy as np
import config
import i18n

try:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    PYGAME_OK = True
except Exception:
    pygame = None
    PYGAME_OK = False


class SoundManager:
    def __init__(self):
        self.ready = False
        self.volume = float(config.DEFAULT_VOLUME)
        self.muted = False
        self.sfx = {}
        self.voice_cache = {}
        self.bgm_cache = {}
        self.current_bgm = None
        self.last_voice_t = 0.0
        self.last_voice_key = ""
        if not PYGAME_OK:
            return
        try:
            pygame.mixer.pre_init(44100, -16, 2, 1024)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(8)
            self.voice_ch = pygame.mixer.Channel(0)
            self.bgm_ch = pygame.mixer.Channel(1)
            self.sfx_ch = pygame.mixer.Channel(2)
            self.ready = True
            self._init_sfx()
        except Exception as e:
            print(f"[AUDIO] Ошибка инициализации звука: {e}")
            self.ready = False

    def _make_synth_sound(self, freqs, dur=0.22, kind="sine", fade=True):
        sr = 44100
        n = int(sr * dur)
        t = np.linspace(0, dur, n, endpoint=False, dtype=np.float32)
        if isinstance(freqs, (int, float)):
            f_arr = np.full_like(t, float(freqs))
        else:
            f_arr = np.linspace(float(freqs[0]), float(freqs[1]), n, dtype=np.float32)
        phase = 2.0 * np.pi * np.cumsum(f_arr) / sr
        if kind == "square":
            wave = np.sign(np.sin(phase)) * 0.45
        elif kind == "chord":
            wave = (np.sin(phase) + 0.6 * np.sin(phase * 1.25) + 0.4 * np.sin(phase * 1.5)) / 2.0
        else:
            wave = np.sin(phase)
        if fade:
            env = np.linspace(1.0, 0.02, n, dtype=np.float32)
            wave = wave * env
        pcm = np.clip(wave * 18000, -32767, 32767).astype(np.int16)
        stereo = np.column_stack((pcm, pcm))
        return pygame.sndarray.make_sound(stereo)

    def _make_hero_bgm(self, hero_id):
        notes_map = {
            "alpamys":   (220, 293.66, 329.63, 440, 392.0, 329.63),
            "tomiris":   (246.94, 293.66, 369.99, 493.88, 440.0, 369.99),
            "spider":    (261.63, 311.13, 392.0, 523.25, 392.0, 311.13),
            "elsa":      (329.63, 392.0, 493.88, 659.25, 587.33, 493.88),
            "ladybug":   (293.66, 369.99, 440.0, 587.33, 493.88, 440.0),
            "iron":      (196.0, 261.63, 329.63, 392.0, 329.63, 261.63),
            "astronaut": (261.63, 329.63, 493.88, 523.25, 392.0, 329.63),
            "fairy":     (349.23, 440.0, 523.25, 698.46, 587.33, 523.25),
        }
        seq = notes_map.get(hero_id, notes_map["alpamys"])
        sr = 44100
        note_dur = 0.36
        parts = []
        for f in seq:
            n = int(sr * note_dur)
            t = np.linspace(0, note_dur, n, endpoint=False, dtype=np.float32)
            env = np.sin(np.pi * t / note_dur) ** 0.6
            w = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 2.0 * t)) * env * 0.25
            parts.append(w)
        full = np.concatenate(parts)
        pcm = np.clip(full * 14000, -32767, 32767).astype(np.int16)
        return pygame.sndarray.make_sound(np.column_stack((pcm, pcm)))

    def _init_sfx(self):
        sfx_dir = os.path.join(config.BASE_DIR, "sounds", "sfx")
        os.makedirs(sfx_dir, exist_ok=True)
        defaults = {
            "beam":     ((880, 320), 0.18, "square"),
            "shield":   ((320, 640), 0.24, "chord"),
            "fly":      ((260, 580), 0.22, "sine"),
            "hit":      ((520, 960), 0.14, "chord"),
            "special":  ((440, 880), 0.32, "chord"),
            "victory":  ((523, 1046), 0.50, "chord"),
            "select":   ((440, 660), 0.15, "sine"),
        }
        for name, (freqs, dur, kind) in defaults.items():
            wav_p = os.path.join(sfx_dir, f"{name}.wav")
            if os.path.exists(wav_p):
                try:
                    self.sfx[name] = pygame.mixer.Sound(wav_p)
                    continue
                except Exception:
                    pass
            self.sfx[name] = self._make_synth_sound(freqs, dur, kind)

    def change_volume(self, delta):
        self.volume = float(np.clip(self.volume + delta, 0.0, 1.0))
        self.muted = (self.volume <= 0.01)
        if self.ready:
            v = 0.0 if self.muted else self.volume
            self.voice_ch.set_volume(min(1.0, v * 1.15))
            self.sfx_ch.set_volume(v * 0.75)
            self.bgm_ch.set_volume(v * 0.32)

    def toggle_mute(self):
        self.muted = not self.muted
        self.change_volume(0.0)

    def play_sfx(self, name):
        if not self.ready or self.muted:
            return
        snd = self.sfx.get(name)
        if snd is not None:
            self.sfx_ch.set_volume(self.volume * 0.75)
            self.sfx_ch.play(snd)

    def play_voice(self, key, force=False, louder=False):
        if not self.ready or self.muted:
            return
        now = time.perf_counter()
        if not force and key == self.last_voice_key and (now - self.last_voice_t) < 2.8:
            return
        if not force and self.voice_ch.get_busy() and (now - self.last_voice_t) < 1.2:
            return
        lang = i18n.get_lang()
        cache_k = (lang, key)
        if cache_k not in self.voice_cache:
            mp3_p = os.path.join(config.BASE_DIR, "sounds", "voice", lang, f"{key}.mp3")
            if not os.path.exists(mp3_p):
                mp3_p = os.path.join(config.BASE_DIR, "sounds", "voice", "kk", f"{key}.mp3")
            if os.path.exists(mp3_p):
                try:
                    self.voice_cache[cache_k] = pygame.mixer.Sound(mp3_p)
                except Exception:
                    self.voice_cache[cache_k] = None
            else:
                self.voice_cache[cache_k] = None
        snd = self.voice_cache.get(cache_k)
        if snd is not None:
            vol = min(1.0, self.volume * (1.25 if louder else 1.05))
            self.voice_ch.set_volume(vol)
            self.voice_ch.play(snd)
            self.last_voice_key = key
            self.last_voice_t = now

    def start_bgm(self, hero_id):
        if not self.ready or self.muted:
            return
        if self.current_bgm == hero_id and self.bgm_ch.get_busy():
            return
        if hero_id not in self.bgm_cache:
            mp3_p = os.path.join(config.BASE_DIR, "sounds", "music", f"{hero_id}.mp3")
            if os.path.exists(mp3_p):
                try:
                    self.bgm_cache[hero_id] = pygame.mixer.Sound(mp3_p)
                except Exception:
                    self.bgm_cache[hero_id] = self._make_hero_bgm(hero_id)
            else:
                self.bgm_cache[hero_id] = self._make_hero_bgm(hero_id)
        snd = self.bgm_cache.get(hero_id)
        if snd is not None:
            self.bgm_ch.set_volume(self.volume * 0.30)
            self.bgm_ch.play(snd, loops=-1)
            self.current_bgm = hero_id

    def stop_bgm(self):
        if self.ready:
            self.bgm_ch.stop()
            self.current_bgm = None


AUDIO = None


def get_audio():
    global AUDIO
    if AUDIO is None:
        AUDIO = SoundManager()
    return AUDIO
