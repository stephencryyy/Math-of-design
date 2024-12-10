import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from matplotlib.widgets import Button

matplotlib.use('TkAgg')

# Функция для генерации случайной звезды
def generate_star():
    # 1. Случайная генерация количества лучей звезды (от 5 до 10)
    num_rays = random.randint(5, 10)

    # Общее количество точек (внешние и внутренние вершины)
    num_points = 2 * num_rays

    # 2. Случайная генерация длины лучей
    # Генерируем длины для внешних и внутренних вершин
    outer_lengths = np.random.uniform(0.7, 1.0, num_rays)
    inner_lengths = np.random.uniform(0.3, 0.6, num_rays)

    # 3. Случайная генерация цветов закраски секторов звезды
    colors = [np.random.rand(3,) for _ in range(num_points)]

    # Вычисляем углы для каждой вершины
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # Чередуем длины для внешних и внутренних точек
    lengths = np.empty(num_points)
    lengths[::2] = outer_lengths  # Чётные индексы — внешние вершины
    lengths[1::2] = inner_lengths  # Нечётные индексы — внутренние вершины

    # Вычисляем координаты x и y для каждой вершины
    x = lengths * np.cos(angles)
    y = lengths * np.sin(angles)

    return num_points, x, y, colors

# Настройка фигуры и осей
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # Оставляем место внизу для кнопки

ax.set_aspect('equal')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis('off')  # Скрываем оси

# Инициализация полигонов
polygons = []

# Функция инициализации анимации
def init():
    return []

# Функция обновления анимации
def animate(i):
    if i < num_points:
        # Создаём треугольник из центра и двух последовательных вершин
        x_coords = [0, x[i], x[(i + 1) % num_points]]
        y_coords = [0, y[i], y[(i + 1) % num_points]]
        polygon = plt.Polygon(list(zip(x_coords, y_coords)), color=colors[i])
        ax.add_patch(polygon)
        polygons.append(polygon)
    return polygons

# Функция для генерации и запуска новой анимации


# Инициализация первой звезды
num_points, x, y, colors = generate_star()

# Создание анимации
ani = animation.FuncAnimation(
    fig, animate, frames=num_points + 1, init_func=init, interval=500, blit=False, repeat=False)

# Добавление кнопки "Regenerate"
ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])  # [left, bottom, width, height]
button = Button(ax_button, 'Regenerate')

def regenerate(event):
    global ani, num_points, x, y, colors, polygons
    # Останавливаем текущую анимацию, если она существует
    if ani is not None:
        ani.event_source.stop()
        ani = None

    # Очищаем ось от предыдущих полигонов
    for poly in polygons:
        poly.remove()
    polygons = []

    # Генерируем новую звезду
    num_points, x, y, colors = generate_star()

    # Обновляем пределы осей при необходимости
    max_length = np.maximum(np.max(np.abs(x)), np.max(np.abs(y)))
    ax.set_xlim(-max_length * 1.1, max_length * 1.1)
    ax.set_ylim(-max_length * 1.1, max_length * 1.1)

    # Запускаем новую анимацию
    ani = animation.FuncAnimation(
        fig, animate, frames=num_points + 1, init_func=init, interval=500, blit=False, repeat=False)
    plt.draw()

# Привязка функции обработчика к кнопке

button.on_clicked(regenerate)

plt.show()
