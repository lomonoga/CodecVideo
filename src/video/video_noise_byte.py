import cv2

video_path = '../../resources/clean/test.mp4'

output_video_path = '../../resources/output/test.mp4'

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
count_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео.")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Конец видео.")
            break

        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        final_frame = cv2.cvtColor(ycrcb_frame, cv2.COLOR_YCrCb2BGR)

        out.write(final_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
