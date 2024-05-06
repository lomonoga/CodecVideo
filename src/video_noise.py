import random
import cv2

# Путь к исходному видеофайлу
video_path = '../resources/clean/test.mp4'

# Путь для сохранения измененного видео
output_video_path = '../resources/output/test.mp4'

# Создание объекта видеозахвата для исходного видео
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
count_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Количество элементов в списке
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Генерируем список случайных целых чисел в заданных границах
random_width = [random.randint(0, frame_width - 1) for _ in range(count_frame + 10)]
random_height = [random.randint(0, frame_height - 1) for _ in range(count_frame + 10)]

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

        # Преобразуем изображение в формат YCbCr
        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)


        frame_width
        frame_height

        start_x, start_y = 100, 100  # Начальная координата x и y верхнего левого угла прямоугольной области
        end_x, end_y = 200, 200  # Конечная координата x и y нижнего правого угла прямоугольной области

        # Задаем новые значения компонент цвета для выбранных пикселей в формате YCrCb
        new_Y, new_Cr, new_Cb = 128, 128, 128  # Пример новых значений

        # Меняем значения компонент цвета выбранных пикселей
        ycrcb_frame[start_y:end_y, start_x:end_x, 0] = new_Y  # Компонента Y
        ycrcb_frame[start_y:end_y, start_x:end_x, 1] = new_Cr  # Компонента Cr
        ycrcb_frame[start_y:end_y, start_x:end_x, 2] = new_Cb  # Компонента Cb



        bgr_frame = cv2.cvtColor(ycrcb_frame, cv2.COLOR_YCrCb2BGR)

        # Запись измененного кадра в видео
        out.write(bgr_frame)

# Освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()
