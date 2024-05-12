import random
import cv2

# Путь к исходному видеофайлу
video_path = '../resources/clean/test.mp4'

# Путь для сохранения измененного видео
output_video_path = '../resources/output/test.mp4'

# Создание объекта видеозахвата для исходного видео
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
count_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Количество элементов в списке
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Ширина кадра
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Высота кадра

# Создание объекта видеозаписи для записи измененного видео
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Проверка, открыт ли поток видео
if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео.")
else:
    while True:
        # Считывание кадра
        ret, frame = cap.read()

        # Проверка, успешно ли считан кадр
        if not ret:
            print("Конец видео.")
            break

        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        final_frame = cv2.cvtColor(ycrcb_frame, cv2.COLOR_YCrCb2BGR)

        out.write(final_frame)

# Освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()
