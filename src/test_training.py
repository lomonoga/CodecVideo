import cv2
import numpy as np
import tensorflow as tf
import video.all_def_noise_video as methods_noise


def load_data(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))
        frames.append(frame)
    cap.release()
    frames = np.array(frames, dtype=np.float32) / 255.0
    return frames, frames


def generate_data(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        noisy_frame = methods_noise.combine_noise_video(frame)
        frame = cv2.resize(frame, (64, 64)) / 255.0
        noisy_frame = cv2.resize(noisy_frame, (64, 64)) / 255.0
        yield np.expand_dims(noisy_frame, axis=0), np.expand_dims(frame, axis=0)
    cap.release()


video_path = '../resources/clean/test.mp4'
x_train, y_train = load_data(video_path)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
])

model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())


model.fit(generate_data(video_path), epochs=10, steps_per_epoch=100)

model.save('../resources/models/video_denoising_autoencoder.h5')
print("Модель сохранена в video_denoising_autoencoder.h5")
