import random

import cv2
import numpy as np


def add_phase_intermodulation(frame: np.ndarray,
                              chance: float = 1,
                              amplitude: int = 30,
                              frequency: int = 5) -> np.ndarray:
    if random.uniform(0.0, 1.0) > chance:
        return frame
    noisy_frame = frame.copy()
    rows, cols, _ = frame.shape
    for i in range(rows):
        offset = int(amplitude * np.sin(2 * np.pi * frequency * i / rows))
        if offset > 0:
            noisy_frame[i, :-offset] = frame[i, offset:]
        elif offset < 0:
            noisy_frame[i, -offset:] = frame[i, :offset]
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

        noise_frame = add_phase_intermodulation(
            frame=ycrcb_frame,
            chance=0.23,
            amplitude=random.randint(10, 40),
            frequency=random.randint(5, 90)
        ).astype(np.uint8)

        final_frame = cv2.cvtColor(noise_frame, cv2.COLOR_YCrCb2BGR)

        out.write(final_frame)

out.release()
