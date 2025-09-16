import cv2
import numpy as np
import os
import argparse
import glob

# Константы для параметров рисования
DEFAULT_COLOR_RECTANGLE = (0, 0, 255)  # Красный цвет в BGR
DEFAULT_COLOR_CIRCLE = (0, 255, 0)     # Зеленый цвет
DEFAULT_COLOR_TEXT = (255, 0, 0)       # Синий цвет
DEFAULT_THICKNESS = 2
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_FONT_SCALE = 0.8
DEFAULT_TEXT = "Sample Text"
DEFAULT_OUTPUT_NAME = "output_image.jpg"

def load_image(image_path):
    """
    Загружает изображение из файла или создаёт чёрный холст, если файл не найден.

    Args:
        image_path (str | None): Путь к входному изображению. Если None или файл не существует,
                                 функция возвращает чёрное изображение стандартного размера.

    Returns:
        numpy.ndarray: Изображение в формате BGR (dtype=np.uint8), shape = (H, W, 3).
    """
    if image_path and os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            return image

    # Создаем черное изображение стандартного размера
    return np.zeros((400, 600, 3), dtype=np.uint8)

def validate_coordinates(image, coordinates, shape_type):
    """
    Проверяет корректность координат для заданного типа фигуры относительно размеров изображения.

    Args:
        image (numpy.ndarray): Изображение (используется для определения ширины и высоты).
        coordinates (tuple): Координаты фигуры.
            - rectangle: (x1, y1, x2, y2)
            - circle: (x, y, radius)
            - text: (x, y)
        shape_type (str): Тип фигуры: 'rectangle', 'circle' или 'text'.

    Returns:
        bool: True, если координаты валидны (внутри границ и соответствуют требованиям типа), иначе False.
    """
    height, width = image.shape[:2]

    if shape_type == 'rectangle':
        x1, y1, x2, y2 = coordinates
        return (0 <= x1 < width and 0 <= y1 < height and
                0 <= x2 < width and 0 <= y2 < height and
                x1 < x2 and y1 < y2)

    elif shape_type == 'circle':
        x, y, radius = coordinates
        return (0 <= x - radius and x + radius < width and
                0 <= y - radius and y + radius < height and
                radius > 0)

    elif shape_type == 'text':
        x, y = coordinates
        return 0 <= x < width and 0 <= y < height

    return False

def draw_rectangle(image, start_point, end_point, color, thickness):
    """
    Рисует прямоугольник на изображении с проверкой координат.

    Args:
        image (numpy.ndarray): Изображение, на котором выполняется рисование (изменяется in-place).
        start_point (tuple[int, int]): Координаты верхнего левого угла (x1, y1).
        end_point (tuple[int, int]): Координаты нижнего правого угла (x2, y2).
        color (tuple[int, int, int]): Цвет прямоугольника в формате BGR.
        thickness (int): Толщина линии. Если thickness == -1 — прямоугольник заливается.

    Returns:
        numpy.ndarray: То же изображение с нанесённым прямоугольником.
    """
    if validate_coordinates(image, (*start_point, *end_point), 'rectangle'):
        cv2.rectangle(image, start_point, end_point, color, thickness)
    else:
        print("Предупреждение: координаты прямоугольника выходят за границы изображения")
    return image

def draw_circle(image, center, radius, color, thickness):
    """"
    Рисует круг на изображении с проверкой координат.

    Args:
        image (numpy.ndarray): Изображение, на котором выполняется рисование (изменяется in-place).
        center (tuple[int, int]): Центр круга (x, y).
        radius (int): Радиус круга в пикселях.
        color (tuple[int, int, int]): Цвет круга в формате BGR.
        thickness (int): Толщина линии. Если thickness == -1 — круг заливается.

    Returns:
        numpy.ndarray: То же изображение с нанесённым кругом.
    """
    if validate_coordinates(image, (*center, radius), 'circle'):
        cv2.circle(image, center, radius, color, thickness)
    else:
        print("Предупреждение: координаты круга выходят за границы изображения")
    return image

def draw_text(image, text, position, font, font_scale, color, thickness, line_type=cv2.LINE_AA):
    """
    Добавляет текст на изображение с проверкой позиции.

    Args:
        image (numpy.ndarray): Изображение, на котором будет добавлен текст (изменяется in-place).
        text (str): Текст для отображения.
        position (tuple[int, int]): Начальная точка текста (x, y). В OpenCV это нижний левый угол текста.
        font (int): Шрифт OpenCV (например, cv2.FONT_HERSHEY_SIMPLEX).
        font_scale (float): Масштаб шрифта.
        color (tuple[int, int, int]): Цвет текста в формате BGR.
        thickness (int): Толщина линий текста.
        line_type (int, optional): Тип линии для отрисовки текста. По умолчанию cv2.LINE_AA.

    Returns:
        numpy.ndarray: То же изображение с добавленным текстом.
    """
    if validate_coordinates(image, position, 'text'):
        cv2.putText(image, text, position, font, font_scale, color, thickness, line_type)
    else:
        print("Предупреждение: координаты текста выходят за границы изображения")
    return image

def process_single_image(input_path, output_path, rect_params, circle_params, text_params):
    """
    Обрабатывает одно изображение: загрузка, рисование фигур/текста и сохранение.

    Args:
        input_path (str | None): Путь к входному изображению. Если None — создаётся чёрный холст.
        output_path (str): Путь для сохранения результата (файл).
        rect_params (tuple): Параметры прямоугольника в формате (start_point, end_point, color, thickness).
        circle_params (tuple): Параметры круга в формате (center, radius, color, thickness).
        text_params (tuple): Параметры текста в формате (text, position, font, font_scale, color, thickness).

    Returns:
        numpy.ndarray: Получившееся изображение (BGR), которое также сохранено в output_path.
    """
    # Загрузка изображения
    image = load_image(input_path)

    # Рисование фигур
    image = draw_rectangle(image, *rect_params)
    image = draw_circle(image, *circle_params)
    image = draw_text(image, *text_params)

    # Сохранение результата
    cv2.imwrite(output_path, image)
    print(f"Обработано: {input_path or 'черное изображение'} -> {output_path}")

    return image

def main():
    """
    Парсинг аргументов командной строки и выбор режима работы:
    - Обработка одного файла
    - Обработка всех изображений в директории
    - Создание и обработка чёрного изображения (если input не указан или не существует)

    Args:
        Нет входных параметров (все параметры берутся из CLI).

    Returns:
        None

    Raises:
        SystemExit: Может быть вызван argparse при некорректных аргументах.
    """
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Рисование фигур на изображении')
    parser.add_argument('--input', type=str, help='Путь к входному изображению или папке с изображениями')
    parser.add_argument('--output', type=str, help='Путь для выходного файла или папки')
    parser.add_argument('--text', type=str, default=DEFAULT_TEXT, help='Текст для отображения')
    args = parser.parse_args()

    # Параметры для рисования
    rect_params = [
        (50, 50),           # start_point
        (200, 200),         # end_point
        DEFAULT_COLOR_RECTANGLE,
        DEFAULT_THICKNESS
    ]

    circle_params = [
        (300, 300),         # center
        50,                 # radius
        DEFAULT_COLOR_CIRCLE,
        DEFAULT_THICKNESS
    ]

    text_params = [
        args.text,          # text
        (50, 300),          # position
        DEFAULT_FONT,
        DEFAULT_FONT_SCALE,
        DEFAULT_COLOR_TEXT,
        DEFAULT_THICKNESS
    ]

    # Определение режима работы (один файл или папка)
    if args.input:
        if os.path.isfile(args.input):
            # Обработка одного файла
            output_path = args.output or DEFAULT_OUTPUT_NAME
            result_image = process_single_image(args.input, output_path, rect_params, circle_params, text_params)

            # Отображение результата
            cv2.imshow('Image with shapes and text', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif os.path.isdir(args.input):
            # Обработка всех изображений в папке
            output_dir = args.output or "output"
            os.makedirs(output_dir, exist_ok=True)

            # Поиск изображений
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_paths = []
            for extension in image_extensions:
                image_paths.extend(glob.glob(os.path.join(args.input, extension)))

            # Обработка каждого изображения
            for i, input_path in enumerate(image_paths):
                output_path = os.path.join(output_dir, f"output_{i}_{os.path.basename(input_path)}")
                process_single_image(input_path, output_path, rect_params, circle_params, text_params)

        else:
            print("Указанный путь не существует. Будет создано черное изображение.")
            output_path = args.output or DEFAULT_OUTPUT_NAME
            result_image = process_single_image(None, output_path, rect_params, circle_params, text_params)

            # Отображение результата
            cv2.imshow('Image with shapes and text', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # Обработка без входного файла (создание черного изображения)
        output_path = args.output or DEFAULT_OUTPUT_NAME
        result_image = process_single_image(None, output_path, rect_params, circle_params, text_params)

        # Отображение результата
        cv2.imshow('Image with shapes and text', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
