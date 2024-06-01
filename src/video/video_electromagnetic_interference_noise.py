import random

import cv2
import numpy as np


def add_electromagnetic_interference(frame: np.ndarray, chance: float = random.uniform(a=0.1, b=0.33),
                                     amplitude: int = random.randint(a=50, b=120)) -> np.ndarray:
    rows, cols, _ = frame.shape
    for i in range(rows):
        if np.random.rand() < chance:
            frame[i, :] = frame[i, :] + amplitude * (random.uniform(a=0.1, b=0.6) - np.random.rand(cols, 1))
    return frame


video_path = '../../resources/clean/test.mp4'
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('../../resources/output/test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео.")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Конец видео.")
            break

        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        noise_frame = add_electromagnetic_interference(
            frame=ycrcb_frame,
            chance=random.uniform(a=0.1, b=0.33),
            amplitude=random.randint(a=50, b=120)
        ).astype(np.uint8)

        final_frame = cv2.cvtColor(noise_frame, cv2.COLOR_YCrCb2BGR)

        out.write(final_frame)

out.release()
