import cv2
import numpy as np


def add_thermal_noise(frame: np.ndarray, mean: float = 1.0, stddev: float = 2.0) -> np.ndarray:
    noise = np.random.normal(loc=mean, scale=stddev, size=frame.shape).astype(np.uint8)
    noisy_frame = cv2.add(frame, noise)
    return noisy_frame


# Загрузка видеофайла
video_path = '../../resources/clean/test.mp4'
cap = cv2.VideoCapture(video_path)

# Считывание параметров видео
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Сохранение видео с шумом в новый файл
out = cv2.VideoWriter('../../resources/output/test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

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

        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        noise_frame = add_thermal_noise(
            frame=ycrcb_frame
        ).astype(np.uint8)

        final_frame = cv2.cvtColor(noise_frame, cv2.COLOR_YCrCb2BGR)

        out.write(final_frame)

out.release()
