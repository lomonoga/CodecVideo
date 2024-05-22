import random
import cv2
import numpy as np


def add_impulse_noise_to_frame(frame: np.ndarray, probability: float, amplitude: float) -> np.ndarray:
    """
    Добавляет импульсные помехи к кадру видео.

    - frame: трехмерный массив NumPy, представляющий кадр
    - probability: вероятность появления импульсной помехи в каждом пикселе каждого кадра.
    - amplitude: амплитуда импульсной помехи.
    """
    noise = np.random.choice(a=[0, amplitude], size=frame.shape, p=[1 - probability, probability])
    noisy_frame = frame + noise

    return noisy_frame


video_path = '../../resources/clean/test.mp4'
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start_probability = 0.2
end_probability = 0.5
start_amplitude = 10
end_amplitude = 255

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

        noise_frame = add_impulse_noise_to_frame(
            frame=ycrcb_frame,
            probability=random.uniform(a=start_probability, b=end_probability),
            amplitude=random.uniform(a=start_amplitude, b=end_amplitude)
        ).astype(np.uint8)

        final_frame = cv2.cvtColor(noise_frame, cv2.COLOR_YCrCb2BGR)

        out.write(final_frame)

out.release()
