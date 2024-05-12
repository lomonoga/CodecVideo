import cv2
import numpy as np


def add_impulse_noise_to_frame(frame, probability, amplitude):
    """
    Добавляет импульсные помехи к кадру видео.

    Параметры:
    - frame: трехмерный массив NumPy, представляющий кадр
    - probability: вероятность появления импульсной помехи в каждом пикселе каждого кадра.
    - amplitude: амплитуда импульсной помехи.
    """
    noise = np.random.choice([0, amplitude], size=frame.shape, p=[1 - probability, probability])
    noisy_frame = frame + noise

    return noisy_frame


# Загрузка видеофайла
video_path = '../resources/clean/test.mp4'
cap = cv2.VideoCapture(video_path)

# Считывание параметров видео
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Добавление импульсных помех
probability = 0.15
amplitude = 5_000

# Сохранение видео с шумом в новый файл
out = cv2.VideoWriter('../resources/output/test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео.")
else:
    while True:
        # Считывание кадра
        ret, frame = cap.read()

        # Проверка, успешно ли считан кадр
        if not ret:
            print("Конец видео.")
            break

        ycrcb_frame = cv2.cvtColor(frame,  cv2.COLOR_BGR2YCrCb)

        noise_frame = add_impulse_noise_to_frame(ycrcb_frame, probability, amplitude).astype(np.uint8)

        final_frame = cv2.cvtColor(noise_frame, cv2.COLOR_YCrCb2BGR)

        out.write(final_frame)

out.release()
