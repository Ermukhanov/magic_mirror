#!/bin/bash
cd "$(dirname "$0")"

echo "=== Сиқырлы Айна (Magic Mirror) ==="
echo "Тексеру... / Проверка окружения..."

if ! command -v python3 &> /dev/null; then
    echo "Python 3 табылмады! / Python 3 не найден!"
    echo "Орнатыңыз: sudo apt update && sudo apt install -y python3 python3-pip"
    exit 1
fi

if ! python3 -c "import cv2, mediapipe, pygame" &> /dev/null; then
    echo "Кітапханалар орнатылуда... / Установка библиотек..."
    pip3 install -r requirements_light.txt
fi

echo "Ойын іске қосылуда... / Запуск игры..."
python3 magic_mirror_v2.py
