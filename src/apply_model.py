import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('../resources/models/video_denoising_autoencoder.h5')


def denoise_frame(input_frame):
    expand_frame = np.expand_dims(input_frame, axis=0)
    return model.predict(expand_frame)[0]


video_path = '../resources/output/combine_noise.mp4'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Ошибка при открытии видео.")
    exit()

ret, first_frame = cap.read()
if not ret:
    print("Ошибка при чтении первого кадра видео.")
    exit()
frame_height, frame_width, _ = first_frame.shape
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = '../resources/applied/denoised_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    denoised_frame = denoise_frame(frame)
    #denoised_frame = np.clip(denoised_frame, 0, 255).astype(np.uint8)
    out.write(denoised_frame)

cap.release()
out.release()

print(f'Улучшенное видео сохранено в {output_path}')
