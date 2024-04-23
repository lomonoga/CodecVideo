import cv2
import numpy as np


def add_noise(frame, intensity=25):
    # Генерация случайного шума для каждого пикселя кадра
    noise = np.random.normal(loc=0, scale=intensity, size=frame.shape).astype(np.uint8)

    # Применение шума к кадру
    noisy_frame = cv2.add(frame, noise)

    return noisy_frame


def add_noise_to_video(input_video, output_video, intensity=25):
    # Открыть видеопоток для чтения
    cap = cv2.VideoCapture(input_video)

    # Получить информацию о видео: ширина, высота, частота кадров и т. д.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Создать объект VideoWriter для записи видео с теми же параметрами, что и у исходного видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Проход по каждому кадру видео
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Наложение шума на кадр
        noisy_frame = add_noise(frame, intensity=intensity)

        # Записать измененный кадр в выходное видео
        out.write(noisy_frame)

    # Освободить ресурсы
    cap.release()
    out.release()


# Пример использования
input_video = 'input_video.mp4'
output_video = 'output_video_with_noise.mp4'

# Наложение шума на видео
add_noise_to_video(input_video, output_video, intensity=25)
