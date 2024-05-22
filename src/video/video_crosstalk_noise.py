import random

import cv2
import numpy as np


def add_crosstalk(frame: np.ndarray, chance: float = 0.13, shift: int = 10) -> np.ndarray:
    noisy_frame = frame.copy()
    if random.uniform(0.0, 1.0) < chance:
        noisy_frame[:, shift:] = frame[:, :-shift] * random.uniform(0.1, 1.0)
        noisy_frame += frame
    return noisy_frame


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

        noise_frame = add_crosstalk(
            frame=ycrcb_frame,
            chance=0.17,
            shift=random.randint(2, 12)
        ).astype(np.uint8)

        final_frame = cv2.cvtColor(noise_frame, cv2.COLOR_YCrCb2BGR)

        out.write(final_frame)

out.release()
