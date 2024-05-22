import random
import cv2

video_path = '../../resources/clean/test.mp4'

output_video_path = '../../resources/output/test.mp4'

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
count_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Количество элементов в списке
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Генерируем список случайных целых чисел в заданных границах
random_count_noise_frame = [random.randint(0, frame_height - 1) for _ in
                            range(random.randint(0, (
                                    frame_width + frame_height) // 48))]  # Менять для уменьшения зашумления

random_width = [[random.randint(0, frame_width - 1) for _ in random_count_noise_frame] for _ in
                range(count_frame + 1)]
random_height = [[random.randint(0, frame_height - 1) for _ in random_count_noise_frame] for _ in
                 range(count_frame + 1)]

random_Y = [random.randint(224, 255) for _ in range(count_frame + 1)]
random_Cr = [random.randint(96, 160) for _ in range(count_frame + 1)]
random_Cb = [random.randint(96, 160) for _ in range(count_frame + 1)]

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео.")
else:
    number_frame = 0
    number_YCrCb = 0
    while True:
        ret, frame = cap.read()
        final_frame = frame

        if not ret:
            print("Конец видео.")
            break

        if random.random() < 0.14:  # Шанс зашумления кадра
            ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

            y_array = random_height[number_frame]
            x_array = random_width[number_frame]

            for index in range(len(y_array)):
                new_Y, new_Cr, new_Cb = (
                    random_Y[number_YCrCb],
                    random_Cr[number_YCrCb],
                    random_Cb[number_YCrCb]
                )

                ycrcb_frame[y_array[index]:random.randint(y_array[index], frame_height),
                x_array[index]:random.randint(x_array[index], frame_width), 0] = new_Y

                ycrcb_frame[y_array[index]:random.randint(y_array[index], frame_height),
                x_array[index]:random.randint(x_array[index], frame_width), 1] = new_Cr

                ycrcb_frame[y_array[index]:random.randint(y_array[index], frame_height),
                x_array[index]:random.randint(x_array[index], frame_width), 2] = new_Cb

            final_frame = cv2.cvtColor(ycrcb_frame, cv2.COLOR_YCrCb2BGR)

        out.write(final_frame)

        number_frame += 1
        number_YCrCb += 1

# Освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()
