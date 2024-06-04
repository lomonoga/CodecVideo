import subprocess

# Пути к исходному и закодированному видео
input_video_path = "../../resources/clean/test.mp4"
output_video_path = "../resources/output/test.mp4"


def encode_video_to_hevc(input_video, output_video):
    # Команда для кодирования видео в формат HEVC с помощью ffmpeg
    command = [
        'ffmpeg',
        '-i', input_video,
        '-c:v', 'hevc',  # Используем кодек HEVC
        '-preset', 'medium',  # Устанавливаем настройку кодирования
        output_video
    ]
    # Выполнение команды
    subprocess.run(command)


# Кодирование видео в формат HEVC
encode_video_to_hevc(input_video_path, output_video_path)
