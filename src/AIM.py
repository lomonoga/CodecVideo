import cv2
import numpy as np


def encode_AIM_video(video_path):
    cap = cv2.VideoCapture(video_path)

    encoded_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразование кадра в одномерный массив байтов
        byte_frame = frame.flatten().tobytes()

        # Кодирование по АИМ
        encoded_frame = bytearray()
        for byte in byte_frame:
            byte_str = bin(byte)[2:].zfill(8)  # Представление каждого байта в виде строки битов
            for bit in byte_str:
                if bit == '1':
                    encoded_frame.extend(b'\x01\x00')
                else:
                    encoded_frame.extend(b'\x00\x01')

        encoded_frames.append(encoded_frame)

    cap.release()
    return encoded_frames


def decode_AIM_video(encoded_frames, frame_size):
    decoded_frames = []
    for encoded_frame in encoded_frames:
        decoded_frame = bytearray()
        for i in range(0, len(encoded_frame), 2):
            if encoded_frame[i] == 1:
                decoded_frame.append(255)
            else:
                decoded_frame.append(0)
        # Преобразование одномерного массива байтов в кадр
        decoded_frame = np.frombuffer(decoded_frame, dtype=np.uint8).reshape(frame_size)
        decoded_frames.append(decoded_frame)
    return decoded_frames


def get_frame_size(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame.shape[:2]  # Возвращает (высота, ширина) кадра


video_path = '../resources/output/ttt.mp4'

# Определение размера кадров в видео
frame_size = get_frame_size(video_path)

# Пример использования
video_path = '../resources/output/ttt.mp4'

# Кодирование видео по АИМ
encoded_frames = encode_AIM_video(video_path)

# Декодирование видео по АИМ
decoded_frames = decode_AIM_video(encoded_frames, frame_size)

# Просмотр результата
for frame in decoded_frames:
    cv2.imshow('Decoded Video', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
