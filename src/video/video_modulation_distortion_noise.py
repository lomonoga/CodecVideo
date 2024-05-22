import random

import cv2
import numpy as np


def modulation_distortion(frame: np.ndarray,
                          chance: float = 1,
                          factor: float = 1.5,
                          additional_value: int = 0) -> np.ndarray:
    return cv2.convertScaleAbs(frame, alpha=factor, beta=additional_value) if \
        (chance > random.uniform(0.0, 1.0)) else frame


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

        noise_frame = modulation_distortion(
            frame=ycrcb_frame,
            chance=0.23,
            factor=random.uniform(0.7, 1.7),
            additional_value=random.randint(-9, 9)
        ).astype(np.uint8)

        final_frame = cv2.cvtColor(noise_frame, cv2.COLOR_YCrCb2BGR)

        out.write(final_frame)

out.release()
