import cv2
import numpy as np


# Загружаем изображение
image = cv2.imread('image.jpg')

# Проверяем, загрузилось ли изображение
if image is None:
    # Создаем черное изображение, если файл не найден
    image = np.zeros((400, 600, 3), dtype=np.uint8)

# Параметры рисования
color_rectangle = (0, 0, 255)  # Красный цвет в BGR
color_circle = (0, 255, 0)     # Зеленый цвет
color_text = (255, 0, 0)       # Синий цвет
thickness = 2                  # Толщина линий

# Рисуем прямоугольник
start_point_rect = (50, 50)
end_point_rect = (200, 200)
cv2.rectangle(image, start_point_rect, end_point_rect, color_rectangle, thickness)

# Рисуем круг
center_circle = (300, 300)
radius = 50
cv2.circle(image, center_circle, radius, color_circle, thickness)

# Добавляем текст
text = "Stupid Man"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
org = (50, 300)
cv2.putText(image, text, org, font, font_scale, color_text, thickness, cv2.LINE_AA)

# Отображаем результат в окне OpenCV
cv2.imshow('Image with shapes and text', image)
cv2.waitKey(0)  # Ждем нажатия любой клавиши
cv2.destroyAllWindows()

# Сохраняем результат
cv2.imwrite('output_image.jpg', image)
