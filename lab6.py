import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import matplotlib

matplotlib.use('TkAgg')

animate = True  # Установите False, если анимация не нужна

num_colors = 4  # Установите 3 или 4

# Выбор типа паркета
print("Выберите тип паркета:")
print("1. Треугольники")
print("2. Четырехугольники")
print("3. Шестиугольник")
print("4. Полу-паркет (звезда и пятиугольник)")

tile_choice = int(input("Введите номер типа паркета (1-4): "))

# Функция для генерации случайных цветов с условием, что соседние плитки не одного цвета
def generate_colors(num_tiles, adjacency_list, num_colors):
    colors = {}
    for tile in range(num_tiles):
        available_colors = set(range(num_colors))
        for neighbor in adjacency_list[tile]:
            if neighbor in colors:
                if colors[neighbor] in available_colors:
                    available_colors.remove(colors[neighbor])
        if available_colors:
            colors[tile] = random.choice(list(available_colors))
        else:
            colors[tile] = random.randint(0, num_colors -1)  # Если нет доступных цветов
    return [colors[i] for i in range(num_tiles)]

# Подготовка фигуры
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.axis('off')

# Список для хранения патчей (плиток)
patches = []
# Список для хранения списков соседей для каждой плитки (для раскраски)
adjacency_list = []

if tile_choice == 1:
    # 2.1 Разнообразные треугольники (3 штуки)
    print("Выберите тип треугольника:")
    print("1. Равносторонний треугольник")
    print("2. Прямоугольный треугольник")
    print("3. Равнобедренный треугольник")
    triangle_choice = int(input("Введите номер треугольника (1-3): "))

    # Создаем сетку треугольников для замощения
    size = 10  # Размер сетки
    num_tiles = 0
    tile_indices = {}
    for i in range(size):
        for j in range(size):
            # Создаем треугольник в зависимости от выбора
            if triangle_choice == 1:
                # Равносторонний треугольник
                side = 1
                h = np.sqrt(3) / 2 * side
                if (i + j) % 2 == 0:
                    points = np.array([
                        [j * side / 2, i * h],
                        [(j + 1) * side / 2, (i + 1) * h],
                        [(j + 2) * side / 2, i * h]
                    ])
                else:
                    points = np.array([
                        [j * side / 2, (i + 1) * h],
                        [(j + 1) * side / 2, i * h],
                        [(j + 2) * side / 2, (i + 1) * h]
                    ])
            elif triangle_choice == 2:
                # Прямоугольный треугольник
                side = 1
                if (i + j) % 2 == 0:
                    points = np.array([
                        [j * side, i * side],
                        [(j + 1) * side, i * side],
                        [j * side, (i + 1) * side]
                    ])
                else:
                    points = np.array([
                        [(j + 1) * side, i * side],
                        [(j + 1) * side, (i + 1) * side],
                        [j * side, (i + 1) * side]
                    ])
            elif triangle_choice == 3:
                # Равнобедренный треугольник
                base = 1
                height = 1
                if (i + j) % 2 == 0:
                    points = np.array([
                        [j * base, i * height],
                        [(j + 1) * base, i * height],
                        [(j + 0.5) * base, (i + 1) * height]
                    ])
                else:
                    points = np.array([
                        [(j + 0.5) * base, i * height],
                        [(j + 1.5) * base, i * height],
                        [(j + 1) * base, (i + 1) * height]
                    ])
            else:
                print("Неверный выбор треугольника.")
                exit()

            polygon = Polygon(points)
            patches.append(polygon)
            tile_indices[(i, j)] = num_tiles
            num_tiles +=1

    # Создаем список смежности для раскраски
    adjacency_list = [[] for _ in range(num_tiles)]
    for i in range(size):
        for j in range(size):
            index = tile_indices[(i, j)]
            neighbors = []
            # Проверяем соседей
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    neighbor_index = tile_indices.get((ni, nj))
                    if neighbor_index is not None:
                        adjacency_list[index].append(neighbor_index)

elif tile_choice == 2:
    # 2.2 Разнообразные четырехугольники
    print("Выберите тип четырехугольника:")
    print("1. Квадрат")
    print("2. Ромб")
    print("3. Прямоугольник")
    print("4. Трапеция")
    print("5. Параллелограмм")
    quad_choice = int(input("Введите номер четырехугольника (1-5): "))

    size = 10
    num_tiles = 0
    tile_indices = {}
    for i in range(size):
        for j in range(size):
            if quad_choice == 1:
                # Квадрат
                side = 1
                points = np.array([
                    [j * side, i * side],
                    [(j +1)* side, i * side],
                    [(j +1)* side, (i +1)* side],
                    [j * side, (i +1)* side]
                ])
            elif quad_choice == 2:
                # Ромб
                side = 1
                h = np.sqrt(3)/2 * side
                points = np.array([
                    [j * side + side/2, i * h],
                    [j * side + side, i * h + h/2],
                    [j * side + side/2, i * h + h],
                    [j * side, i * h + h/2]
                ])
            elif quad_choice == 3:
                # Прямоугольник
                width = 1
                height = 0.5
                points = np.array([
                    [j * width, i * height],
                    [(j +1)* width, i * height],
                    [(j +1)* width, (i +1)* height],
                    [j * width, (i +1)* height]
                ])
            elif quad_choice == 4:
                # Трапеция
                base = 1
                top = 0.5
                height = 1
                points = np.array([
                    [j * base, i * height],
                    [j * base + base, i * height],
                    [j * base + base - (base - top)/2, (i +1)* height],
                    [j * base + (base - top)/2, (i +1)* height]
                ])
            elif quad_choice ==5:
                # Параллелограмм
                base =1
                side =1
                shift =0.5
                points = np.array([
                    [j * base + shift * (i%2), i * side],
                    [(j +1)* base + shift * (i%2), i * side],
                    [(j +1)* base + shift * (i%2), (i +1)* side],
                    [j * base + shift * (i%2), (i +1)* side]
                ])
            else:
                print("Неверный выбор четырехугольника.")
                exit()

            polygon = Polygon(points)
            patches.append(polygon)
            tile_indices[(i, j)] = num_tiles
            num_tiles +=1

    # Создаем список смежности для раскраски
    adjacency_list = [[] for _ in range(num_tiles)]
    for i in range(size):
        for j in range(size):
            index = tile_indices[(i, j)]
            neighbors = []
            # Проверяем соседей
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    neighbor_index = tile_indices.get((ni, nj))
                    if neighbor_index is not None:
                        adjacency_list[index].append(neighbor_index)

elif tile_choice == 3:
    # 2.3 Один шестиугольник
    size = 10
    num_tiles = 0
    tile_indices = {}
    for i in range(size):
        for j in range(size):
            # Шестиугольник
            side =1
            h = np.sqrt(3)/2 * side
            x_offset = (j + 0.5 * (i %2)) * side * 1.5
            y_offset = i * h * 2

            points = np.array([
                [x_offset + side * np.cos(np.pi/3 * k) for k in range(6)],
                [y_offset + side * np.sin(np.pi/3 * k) for k in range(6)]
            ]).T

            polygon = Polygon(points)
            patches.append(polygon)
            tile_indices[(i, j)] = num_tiles
            num_tiles +=1

    # Создаем список смежности для раскраски
    adjacency_list = [[] for _ in range(num_tiles)]
    for i in range(size):
        for j in range(size):
            index = tile_indices[(i, j)]
            neighbors = []
            # Проверяем соседей (6 направлений)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    neighbor_index = tile_indices.get((ni, nj))
                    if neighbor_index is not None:
                        adjacency_list[index].append(neighbor_index)

elif tile_choice == 4:
    # 3. Полу-паркет – звезда вкупе с пятиугольником (5 б.)
    size = 5  # Размер сетки
    num_tiles = 0
    tile_indices = {}
    for i in range(size):
        for j in range(size):
            center_x = j * 2
            center_y = i * 2

            # Рисуем пятиугольник
            pentagon = np.array([
                [center_x + np.cos(2 * np.pi * k /5), center_y + np.sin(2 * np.pi * k /5)]
                for k in range(5)
            ])
            polygon1 = Polygon(pentagon)
            patches.append(polygon1)
            tile_indices[(i, j, 'pentagon')] = num_tiles
            num_tiles +=1

            # Рисуем звезду
            star = []
            for k in range(10):
                angle = 2 * np.pi * k /10
                radius = 1 if k %2 ==0 else 0.5
                star.append([center_x + radius * np.cos(angle), center_y + radius * np.sin(angle)])
            star = np.array(star)
            polygon2 = Polygon(star)
            patches.append(polygon2)
            tile_indices[(i, j, 'star')] = num_tiles
            num_tiles +=1

    # Создаем список смежности для раскраски
    adjacency_list = [[] for _ in range(num_tiles)]
    # Для каждой фигуры добавляем соседей
    for idx in range(0, num_tiles, 2):
        adjacency_list[idx].append(idx +1)
        adjacency_list[idx +1].append(idx)
        if idx +2 < num_tiles:
            adjacency_list[idx].append(idx +2)
            adjacency_list[idx +1].append(idx +2)
else:
    print("Неверный выбор типа паркета.")
    exit()

# Генерируем цвета для плиток
colors = generate_colors(num_tiles, adjacency_list, num_colors)
cmap = ListedColormap(np.random.rand(num_colors, 3))

# Создаем коллекцию патчей
collection = PatchCollection(patches, cmap=cmap, edgecolor='black')
collection.set_array(np.array(colors))

ax.add_collection(collection)

# Настраиваем пределы осей
ax.autoscale_view()

# Анимация
if animate:
    def init():
        collection.set_array(np.zeros(num_tiles))
        return collection,

    def animate_func(i):
        current_colors = colors[:i]
        collection.set_array(np.array(current_colors + [0]*(num_tiles - len(current_colors))))
        return collection,

    ani = animation.FuncAnimation(fig, animate_func, frames=num_tiles +1, init_func=init, interval=100, blit=True)
    plt.show()
else:
    plt.show()
