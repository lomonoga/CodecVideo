encoded_video_path = "../resources/output/test.mp4"


def read_video_to_bytes(video_path):
    with open(video_path, 'rb') as file:
        video_bytes = file.read()
    return video_bytes


# Чтение закодированного видео в байтовом виде
encoded_video_bytes = read_video_to_bytes(encoded_video_path)

# Сохранение закодированного видео в новый файл MP4
with open('encoded_video_bytes.mp4', 'wb') as file:
    file.write(encoded_video_bytes)
