import cv2
import numpy as np
import tensorflow as tf
import video.all_def_noise_video as methods_noise


def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def generate_data(video_path):
    video_frames = load_video_frames(video_path)
    for frame in video_frames:
        noisy_frame = methods_noise.combine_noise_video(frame)
        yield np.expand_dims(noisy_frame, axis=0), np.expand_dims(frame, axis=0)


def create_model():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(None, None, 3)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])


mse_loss = tf.keras.losses.MeanSquaredError()


@tf.function
def train_step(model, noisy_frame, frame):
    with tf.GradientTape() as tape:
        predictions = model(noisy_frame, training=True)
        loss = mse_loss(frame, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train_model(model, video_path, epochs=10, steps_per_epoch=100):
    video_frames_generator = generate_data(video_path)
    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}/{epochs}')
        for _ in range(steps_per_epoch):
            try:
                noisy_frame, frame = next(video_frames_generator)
            except StopIteration:
                video_frames_generator = generate_data(video_path)
                noisy_frame, frame = next(video_frames_generator)
            train_step(model, noisy_frame, frame)
        print(f'Epoch {epoch + 1}/{epochs} completed')


clean_video_path = '../resources/clean/test.mp4'

model = create_model()
model.compile(optimizer='adam', loss='mean_squared_error')

train_model(model=model, video_path=clean_video_path, epochs=5, steps_per_epoch=30)

model.save('../resources/models/video_denoising_model.h5')
print("Model saved to video_denoising_model.h5")
