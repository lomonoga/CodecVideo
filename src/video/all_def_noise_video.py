import random as rnd
import cv2
import numpy as np


def add_crosstalk(
        frame: np.ndarray,
        shift: int = rnd.randint(2, 12)
) -> np.ndarray:
    noisy_frame = frame.copy()
    noisy_frame[:, shift:] = frame[:, :-shift] * rnd.uniform(0.1, 1.0)
    noisy_frame += frame

    return noisy_frame


def add_electromagnetic_interference(
        frame: np.ndarray,
        amplitude: int = rnd.randint(a=50, b=120)
) -> np.ndarray:
    noisy_frame = frame.copy()
    rows, cols, _ = noisy_frame.shape
    for i in range(rows):
        noisy_frame[i, :] = noisy_frame[i, :] + amplitude * (rnd.uniform(a=0.1, b=0.6) - np.random.rand(cols, 1))

    return noisy_frame


def add_impulse_noise_to_frame(
        frame: np.ndarray,
        probability: float = rnd.uniform(a=0.2, b=0.5),
        amplitude: float = rnd.uniform(a=10, b=255)
) -> np.ndarray:
    noise = np.random.choice(a=[0, amplitude], size=frame.shape, p=[1 - probability, probability])
    noisy_frame = frame + noise

    return noisy_frame


def modulation_distortion(
        frame: np.ndarray,
        factor: float = rnd.uniform(0.7, 1.7),
        additional_value: int = rnd.randint(-9, 9)
) -> np.ndarray:
    return cv2.convertScaleAbs(frame, alpha=factor, beta=additional_value)


def add_phase_intermodulation(
        frame: np.ndarray,
        amplitude: int = rnd.randint(10, 40),
        frequency: int = rnd.randint(5, 90)
) -> np.ndarray:
    noisy_frame = frame.copy()
    rows, cols, _ = frame.shape
    for i in range(rows):
        offset = int(amplitude * np.sin(2 * np.pi * frequency * i / rows))
        if offset > 0:
            noisy_frame[i, :-offset] = frame[i, offset:]
        elif offset < 0:
            noisy_frame[i, -offset:] = frame[i, :offset]
    return noisy_frame


def add_thermal_noise(
        frame: np.ndarray,
        mean: float = 1.0,
        stddev: float = 2.0
) -> np.ndarray:
    noise = np.random.normal(loc=mean, scale=stddev, size=frame.shape).astype(np.float32)
    frame_float = frame.astype(np.float32)
    noisy_frame = np.clip(cv2.add(frame_float, noise), 0, 255).astype(np.uint8)

    return noisy_frame


ALL_METHODS = {add_crosstalk: 0.2,
               add_electromagnetic_interference: 0.2,
               add_impulse_noise_to_frame: 0.05,
               modulation_distortion: 0.1,
               add_phase_intermodulation: 0.05,
               add_thermal_noise: 0.05}


def combine_noise_video(frame: np.ndarray) -> np.ndarray:
    changeable_frame = frame.copy()
    for method, probability in ALL_METHODS.items():
        if rnd.uniform(0.0, 1.0) <= probability:
            changeable_frame = method(changeable_frame)

    return changeable_frame
