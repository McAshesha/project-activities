import argparse
import glob
import os
from typing import Optional

import cv2
import numpy as np

# Константы для параметров размытия
DEFAULT_KERNEL_SIZE = 15
DEFAULT_DIRECTION = 'horizontal'
DEFAULT_OUTPUT_NAME = "motion_blur_output.jpg"

def load_image(image_path: Optional[str]) -> np.ndarray:
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

def create_motion_blur_kernel(kernel_size: int, direction: str = 'horizontal') -> np.ndarray:
    """
    Создает ядро для размытия в движении в заданном направлении.

    Args:
        kernel_size (int): Размер ядра (должен быть нечетным).
        direction (str): Направление размытия ('horizontal' или 'vertical').

    Returns:
        numpy.ndarray: Нормализованное ядро свертки для motion blur.

    Raises:
        ValueError: Если kernel_size не является нечетным числом.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size должен быть нечетным числом")

    kernel = np.zeros((kernel_size, kernel_size))

    if direction == 'horizontal':
        # Горизонтальное размытие
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    elif direction == 'vertical':
        # Вертикальное размытие
        kernel[:, int((kernel_size-1)/2)] = np.ones(kernel_size)
    else:
        raise ValueError("Направление должно быть 'horizontal' или 'vertical'")

    # Нормализуем ядро
    kernel /= kernel_size
    return kernel

def apply_motion_blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Применяет эффект motion blur к изображению с помощью свертки.

    Args:
        image (numpy.ndarray): Входное изображение.
        kernel (numpy.ndarray): Ядро свертки для размытия.

    Returns:
        numpy.ndarray: Изображение с примененным эффектом motion blur.
    """
    return cv2.filter2D(image, -1, kernel)

def process_single_image(input_path: Optional[str], output_path: str,
                         kernel_size: int, direction: str) -> np.ndarray:
    """
    Обрабатывает одно изображение: загрузка, применение motion blur и сохранение.

    Args:
        input_path (str | None): Путь к входному изображению. Если None — создаётся чёрный холст.
        output_path (str): Путь для сохранения результата.
        kernel_size (int): Размер ядра для размытия.
        direction (str): Направление размытия ('horizontal' или 'vertical').

    Returns:
        numpy.ndarray: Получившееся изображение с эффектом motion blur.
    """
    # Загрузка изображения
    image = load_image(input_path)

    # Создание ядра размытия
    kernel = create_motion_blur_kernel(kernel_size, direction)

    # Применение эффекта motion blur
    blurred_image = apply_motion_blur(image, kernel)

    # Сохранение результата
    cv2.imwrite(output_path, blurred_image)
    print(f"Обработано: {input_path or 'черное изображение'} -> {output_path}")

    return blurred_image

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
    """
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Создание эффекта motion blur на изображениях')
    parser.add_argument('--input', type=str, help='Путь к входному изображению или папке с изображениями')
    parser.add_argument('--output', type=str, help='Путь для выходного файла или папки')
    parser.add_argument('--kernel_size', type=int, default=DEFAULT_KERNEL_SIZE,
                        help=f'Размер ядра размытия (нечетное число, по умолчанию {DEFAULT_KERNEL_SIZE})')
    parser.add_argument('--direction', type=str, default=DEFAULT_DIRECTION,
                        choices=['horizontal', 'vertical'],
                        help=f'Направление размытия (по умолчанию {DEFAULT_DIRECTION})')

    args = parser.parse_args()

    # Проверка корректности размера ядра
    if args.kernel_size % 2 == 0:
        print("Ошибка: kernel_size должен быть нечетным числом")
        return

    # Определение режима работы (один файл или папка)
    if args.input:
        if os.path.isfile(args.input):
            # Обработка одного файла
            output_path = args.output or DEFAULT_OUTPUT_NAME
            result_image = process_single_image(args.input, output_path,
                                                args.kernel_size, args.direction)

            # Отображение результата
            cv2.imshow('Original', load_image(args.input))
            cv2.imshow('Motion Blur', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif os.path.isdir(args.input):
            # Обработка всех изображений в папке
            output_dir = args.output or "motion_blur_output"
            os.makedirs(output_dir, exist_ok=True)

            # Поиск изображений
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_paths = []
            for extension in image_extensions:
                image_paths.extend(glob.glob(os.path.join(args.input, extension)))

            # Обработка каждого изображения
            for input_path in image_paths:
                output_filename = f"motion_blur_{os.path.basename(input_path)}"
                output_path = os.path.join(output_dir, output_filename)
                process_single_image(input_path, output_path,
                                     args.kernel_size, args.direction)

        else:
            print("Указанный путь не существует. Будет создано черное изображение.")
            output_path = args.output or DEFAULT_OUTPUT_NAME
            result_image = process_single_image(None, output_path,
                                                args.kernel_size, args.direction)

            # Отображение результата
            cv2.imshow('Motion Blur on Black Canvas', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # Обработка без входного файла (создание черного изображения)
        output_path = args.output or DEFAULT_OUTPUT_NAME
        result_image = process_single_image(None, output_path,
                                            args.kernel_size, args.direction)

        # Отображение результата
        cv2.imshow('Motion Blur on Black Canvas', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
