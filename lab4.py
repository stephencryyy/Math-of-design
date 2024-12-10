import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import random

matplotlib.use('TkAgg')  # Если требуется, можно изменить backend


class KochSnowflakeAnimator:
    def __init__(self, max_iter=4, sides=3, angle_deg=60, color='blue'):
        """
        max_iter: количество итераций (глубина снежинки)
        sides: количество сторон начального многоугольника
        angle_deg: базовый угол для построения (для классической снежинки Коха = 60)
        color: цвет линии
        """
        self.max_iter = max_iter
        self.sides = sides
        self.angle = np.deg2rad(angle_deg)  # переводим градусы в радианы
        self.color = color

        # Генерируем L-систему для снежинки
        # Аксима: F -- F -- F (для треугольника),
        # но мы можем начать с n-угольника: F--F--F--... (n раз), где угол = 360/n
        self.axiom = self.generate_polygon_axiom(self.sides)
        self.rules = {'F': 'F+F--F+F'}

        # Для удобства создадим список строк для каждой итерации
        self.strings = [self.axiom]
        for i in range(self.max_iter):
            new_s = ""
            for c in self.strings[-1]:
                if c in self.rules:
                    new_s += self.rules[c]
                else:
                    new_s += c
            self.strings.append(new_s)

        # Генерируем координаты для каждой итерации
        self.iter_data = []
        for s in self.strings:
            x, y = self.string_to_coords(s)
            self.iter_data.append((x, y))

        # Настраиваем фигуру
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.line, = self.ax.plot([], [], color=self.color, lw=1)

    def generate_polygon_axiom(self, n):

        polygon_angle = 360.0 / n
        turns = round(polygon_angle / np.degrees(self.angle))  # число '-' для поворота

        # Создаём аксиому: n раз "F" + поворот, где поворот = turns раз '-'
        segment = "F" + "-" * turns
        axiom = segment * n
        return axiom

    def string_to_coords(self, s):
        x, y = [0], [0]
        angle = 0
        step = 1.0
        for c in s:
            if c == 'F':
                x.append(x[-1] + step * np.cos(angle))
                y.append(y[-1] + step * np.sin(angle))
            elif c == '+':
                angle += self.angle
            elif c == '-':
                angle -= self.angle
        return np.array(x), np.array(y)

    def init_plot(self):
        self.ax.set_aspect('equal')
        self.ax.set_title("Снежинка Коха. Итерация: 0")
        self.line.set_data([], [])
        return self.line,

    def update_frame(self, frame):
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_title(f"Снежинка Коха. Итерация: {frame}")
        x, y = self.iter_data[frame]
        self.ax.plot(x, y, color=self.color, lw=1)
        return self.line,

    def animate(self):
        frames = self.max_iter + 1
        ani = FuncAnimation(self.fig, self.update_frame, frames=frames,
                            init_func=self.init_plot, interval=1000, blit=False, repeat=False)
        plt.show()


if __name__ == "__main__":
    # Случайная генерация количества лучей (число сторон)
    sides = random.randint(3, 6)  # от 3 до 6 сторон
    # Случайный цвет
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
    color = random.choice(colors)
    # Случайная глубина (кол-во итераций)
    max_iter = random.randint(2, 10)
    # Случайный угол наклона (в градусах) - для Коха классика = 60°, но пусть будет вариация
    angle_deg = random.choice([30, 60, 45])  # Можно добавить больше вариантов

    print(f"Случайные параметры:")
    print(f"Количество сторон: {sides}")
    print(f"Цвет: {color}")
    print(f"Глубина (итерации): {max_iter}")
    print(f"Угол: {angle_deg}°")

    animator = KochSnowflakeAnimator(max_iter=max_iter, sides=sides, angle_deg=angle_deg, color=color)
    animator.animate()
