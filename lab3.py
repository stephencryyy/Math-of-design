import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # Например, если доступен Tkinter

def get_param(prompt, default, cast_func):
    """Запрашивает у пользователя параметр. Если ввод пустой - возвращает default."""
    val = input(prompt + f" (нажмите Enter для значения по умолчанию: {default}): ")
    if val.strip() == "":
        return default
    return cast_func(val)

class MandelbrotAnimator:
    def __init__(self, start_iter=10, end_iter=200, step=10, resolution=300, zoom=1.0, center=(-0.5, 0)):
        """
        start_iter, end_iter - изменение максимального числа итераций
        step - шаг увеличения итераций на кадр
        resolution - разрешение картинки
        zoom - масштаб
        center - центр области (кортеж (x,y))
        """
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.step = step
        self.resolution = resolution
        self.zoom = zoom
        self.center = center

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.im = self.ax.imshow(np.zeros((resolution, resolution)),
                                 extent=[-2, 1, -1.5, 1.5], cmap='magma', origin='lower')
        self.ax.set_title(f"Мандельброт. max_iter={self.start_iter}")
        self.fig.colorbar(self.im, ax=self.ax, label='Итерации')

    def mandelbrot_set(self, max_iter, bound=2, resolution=500, center=(-0.5, 0), zoom=1.0):
        x_min, x_max = center[0] - 1.5 * zoom, center[0] + 1.5 * zoom
        y_min, y_max = center[1] - 1.5 * zoom, center[1] + 1.5 * zoom

        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)

        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        Z = np.zeros_like(C)
        M = np.zeros(C.shape, dtype=int)

        for i in range(max_iter):
            mask = (np.abs(Z) <= bound)
            Z[mask] = Z[mask] * Z[mask] + C[mask]
            M[mask & (np.abs(Z) > bound)] = i
        return M

    def init_plot(self):
        M = self.mandelbrot_set(self.start_iter, resolution=self.resolution, center=self.center, zoom=self.zoom)
        self.im.set_data(M)
        self.ax.set_title(f"Мандельброт. max_iter={self.start_iter}")
        return self.im,

    def update_frame(self, frame):
        current_iter = self.start_iter + frame * self.step
        if current_iter > self.end_iter:
            current_iter = self.end_iter
        M = self.mandelbrot_set(current_iter, resolution=self.resolution, center=self.center, zoom=self.zoom)
        self.im.set_data(M)
        self.ax.set_title(f"Мандельброт. max_iter={current_iter}")
        return self.im,

    def animate(self):
        frames = (self.end_iter - self.start_iter) // self.step + 1
        ani = FuncAnimation(self.fig, self.update_frame, frames=frames,
                            init_func=self.init_plot, interval=500, blit=False, repeat=False)
        plt.show()

class SierpinskiAnimator:
    def __init__(self, max_iter=5):
        """
        max_iter - максимальное число итераций.
        На итерации 0 - один большой треугольник.
        На итерации n - 3^n маленьких треугольников.
        """
        self.max_iter = max_iter
        # Генерируем треугольники для всех итераций сразу
        self.iterations = self.generate_iterations(self.max_iter)

        # Создаём фигуру и ось
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.ax.set_aspect('equal')
        self.collection = []
        # Создадим графический объект, который будем обновлять
        # Рисовать будем с помощью fill (заливая каждый треугольник)
        self.patches = []

    def initial_triangle(self):
        # Равносторонний треугольник можно задать координатами:
        # Возьмём за основу координаты, как в "хаотической игре":
        # нижняя сторона от (0,0) до (1,0), верхняя вершина (0.5, sqrt(3)/2)
        return [(0.0,0.0), (1.0,0.0), (0.5, np.sqrt(3)/2)]

    def subdivide_triangle(self, triangle):
        """
        На вход подаётся список из трёх точек [(x1,y1),(x2,y2),(x3,y3)].
        Возвращаем 3 новых треугольника (угловых), исключая центральный.
        """
        (x1,y1),(x2,y2),(x3,y3) = triangle

        # Середины сторон
        x12, y12 = (x1+x2)/2, (y1+y2)/2
        x23, y23 = (x2+x3)/2, (y2+y3)/2
        x31, y31 = (x3+x1)/2, (y3+y1)/2

        # Новый уровень треугольников:
        # Левый нижний
        t1 = [(x1,y1),(x12,y12),(x31,y31)]
        # Правый нижний
        t2 = [(x2,y2),(x23,y23),(x12,y12)]
        # Верхний
        t3 = [(x3,y3),(x31,y31),(x23,y23)]

        return [t1,t2,t3]

    def generate_iterations(self, max_iter):
        """
        Генерируем список итераций.
        iterations[i] - список треугольников на i-й итерации.
        i=0: один треугольник
        i=1: 3 треугольника
        и так далее...
        """
        iterations = []
        # Итерация 0: один большой треугольник
        current = [self.initial_triangle()]
        iterations.append(current)

        for i in range(1, max_iter+1):
            next_level = []
            for tri in current:
                subtris = self.subdivide_triangle(tri)
                next_level.extend(subtris)
            iterations.append(next_level)
            current = next_level
        return iterations

    def init_plot(self):
        # Изначально показываем итерацию 0
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-0.1,1.1)
        self.ax.set_ylim(-0.1,1.0)
        self.ax.set_title("Треугольник Серпинского. Итерация: 0")

        # Рисуем начальный треугольник
        tris = self.iterations[0]
        for tri in tris:
            x = [p[0] for p in tri]
            y = [p[1] for p in tri]
            self.ax.fill(x, y, 'k')
        return []

    def update_frame(self, frame):
        # frame - номер итерации
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-0.1,1.1)
        self.ax.set_ylim(-0.1,1.0)
        self.ax.set_title(f"Треугольник Серпинского. Итерация: {frame}")

        tris = self.iterations[frame]
        for tri in tris:
            x = [p[0] for p in tri]
            y = [p[1] for p in tri]
            self.ax.fill(x, y, 'k')
        return []

    def animate(self):
        frames = self.max_iter + 1  # от 0 до max_iter включительно
        ani = FuncAnimation(self.fig, self.update_frame, frames=frames,
                            init_func=self.init_plot, interval=1000, blit=False, repeat=False)
        plt.show()

class KochAnimator:
    def __init__(self, max_iter=5):
        self.max_iter = max_iter
        self.data = []
        for i in range(self.max_iter + 1):
            x, y = self.koch_curve(i)
            self.data.append((x, y))

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.line, = self.ax.plot([], [], 'b-', lw=1)

    def koch_curve(self, iterations):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.0])

        for _ in range(iterations):
            x_new = []
            y_new = []
            for i in range(len(x) - 1):
                x1, y1 = x[i], y[i]
                x2, y2 = x[i + 1], y[i + 1]

                dx = x2 - x1
                dy = y2 - y1

                xA = x1 + dx / 3
                yA = y1 + dy / 3
                xB = x1 + 2 * dx / 3
                yB = y1 + 2 * dy / 3

                angle = np.pi / 3
                xC = xA + (dx / 3) * np.cos(angle) - (dy / 3) * np.sin(angle)
                yC = yA + (dx / 3) * np.sin(angle) + (dy / 3) * np.cos(angle)

                x_new.extend([x1, xA, xC, xB])
                y_new.extend([y1, yA, yC, yB])
            x_new.append(x[-1])
            y_new.append(y[-1])
            x = np.array(x_new)
            y = np.array(y_new)
        return x, y

    def init_plot(self):
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.5, 0.6)
        self.ax.set_aspect('equal')
        self.ax.set_title("Кривая Коха. Итерация: 0")
        self.line.set_data([], [])
        return self.line,

    def update_frame(self, frame):
        x, y = self.data[frame]
        self.line.set_data(x, y)
        self.ax.set_title(f"Кривая Коха. Итерация: {frame}")
        return self.line,

    def animate(self):
        ani = FuncAnimation(self.fig, self.update_frame, frames=len(self.data),
                            init_func=self.init_plot, interval=1000, repeat=False, blit=False)
        plt.show()

class JuliaAnimator:
    def __init__(self, c=(-0.7, 0.27015), start_iter=10, end_iter=200, step=10, resolution=300, zoom=1.0):
        """
        c - параметр множества Жюлиа (комплексное число)
        start_iter, end_iter, step - аналогично Мандельброту
        resolution - разрешение
        zoom - масштаб
        """
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.step = step
        self.resolution = resolution
        self.zoom = zoom
        self.c = c[0] + 1j * c[1]

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.im = self.ax.imshow(np.zeros((resolution, resolution)),
                                 extent=[-1.5, 1.5, -1.5, 1.5], cmap='magma', origin='lower')
        self.ax.set_title(f"Жюлиа. max_iter={self.start_iter}, c={self.c}")
        self.fig.colorbar(self.im, ax=self.ax, label='Итерации')

    def julia_set(self, c, max_iter=100, bound=2, resolution=500, zoom=1.0):
        x_min, x_max = -1.5 * zoom, 1.5 * zoom
        y_min, y_max = -1.5 * zoom, 1.5 * zoom

        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)

        Z = X + 1j * Y
        M = np.zeros(Z.shape, dtype=int)

        for i in range(max_iter):
            mask = (np.abs(Z) <= bound)
            Z[mask] = Z[mask]*Z[mask] + c
            M[mask & (np.abs(Z) > bound)] = i
        return M

    def init_plot(self):
        M = self.julia_set(self.c, max_iter=self.start_iter, resolution=self.resolution, zoom=self.zoom)
        self.im.set_data(M)
        self.ax.set_title(f"Жюлиа. max_iter={self.start_iter}, c={self.c}")
        return self.im,

    def update_frame(self, frame):
        current_iter = self.start_iter + frame * self.step
        if current_iter > self.end_iter:
            current_iter = self.end_iter
        M = self.julia_set(self.c, max_iter=current_iter, resolution=self.resolution, zoom=self.zoom)
        self.im.set_data(M)
        self.ax.set_title(f"Жюлиа. max_iter={current_iter}, c={self.c}")
        return self.im,

    def animate(self):
        frames = (self.end_iter - self.start_iter) // self.step + 1
        ani = FuncAnimation(self.fig, self.update_frame, frames=frames,
                            init_func=self.init_plot, interval=500, blit=False, repeat=False)
        plt.show()


class HeighwayDragonAnimator:
    def __init__(self, max_iter=10):
        """
        max_iter - максимальное число итераций.
        Аксима: F
        Правила:
        F -> F+G
        G -> F-G
        Угол 90°
        """
        self.max_iter = max_iter
        self.angle = np.pi/2
        self.strings = self.generate_strings(max_iter)
        self.iter_data = []
        for s in self.strings:
            x, y = self.string_to_coords(s)
            self.iter_data.append((x, y))

        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.line, = self.ax.plot([], [], 'r-', lw=1)

    def generate_strings(self, max_iter):
        # Генерируем строки для каждой итерации
        # Аксима: F
        # F -> F+G
        # G -> F-G
        s = "F"
        strings = [s]
        for _ in range(max_iter):
            new_s = ""
            for c in s:
                if c == 'F':
                    new_s += "F+G"
                elif c == 'G':
                    new_s += "F-G"
                else:
                    new_s += c
            s = new_s
            strings.append(s)
        return strings

    def string_to_coords(self, s):
        x, y = [0], [0]
        angle = 0
        step = 1.0
        for c in s:
            if c in ['F','G']:
                x.append(x[-1] + step*np.cos(angle))
                y.append(y[-1] + step*np.sin(angle))
            elif c == '+':
                angle += self.angle
            elif c == '-':
                angle -= self.angle
        return np.array(x), np.array(y)

    def init_plot(self):
        self.ax.set_aspect('equal')
        self.ax.set_title("Дракон Хартера-Хейтуэя. Итерация: 0")
        self.line.set_data([], [])
        return self.line,

    def update_frame(self, frame):
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_title(f"Дракон Хартера-Хейтуэя. Итерация: {frame}")
        x, y = self.iter_data[frame]
        self.ax.plot(x, y, 'r-', lw=1)
        return self.line,

    def animate(self):
        frames = self.max_iter+1
        ani = FuncAnimation(self.fig, self.update_frame, frames=frames,
                            init_func=self.init_plot, interval=1000, blit=False, repeat=False)
        plt.show()

class LeviAnimator:
    def __init__(self, max_iter=10, angle_degrees=45):
        """
        max_iter - максимальное число итераций.
        angle_degrees - угол поворота в градусах (по умолчанию 45°),
                        можно изменять для получения вариаций кривой Леви.

        Правила Л-системы для кривой Леви:
        Аксима: F
        Правило: F -> +F--F+
        """
        self.max_iter = max_iter
        self.angle = np.deg2rad(angle_degrees)  # переводим градусы в радианы
        # Генерируем L-систему для каждой итерации
        self.strings = self.generate_strings(self.max_iter)
        # Для каждой строки генерируем координаты
        self.iter_data = []
        for s in self.strings:
            x, y = self.string_to_coords(s)
            self.iter_data.append((x, y))

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.line, = self.ax.plot([], [], 'b-', lw=1)

    def generate_strings(self, max_iter):
        # Чтобы получить поэтапную генерацию:
        # На итерации 0: "F"
        # На итерации 1: "+F--F+"
        # На итерации 2: применяем правило к каждой F в строке итерации 1 и т.д.
        s = "F"
        strings = [s]
        for _ in range(max_iter):
            new_s = ""
            for c in s:
                if c == 'F':
                    new_s += "+F--F+"
                else:
                    new_s += c
            s = new_s
            strings.append(s)
        return strings

    def string_to_coords(self, s):
        x, y = [0], [0]
        angle = 0
        step = 1.0
        for c in s:
            if c == 'F':
                # шаг вперёд
                x.append(x[-1] + step * np.cos(angle))
                y.append(y[-1] + step * np.sin(angle))
            elif c == '+':
                angle += self.angle
            elif c == '-':
                angle -= self.angle
        return np.array(x), np.array(y)

    def init_plot(self):
        self.ax.set_aspect('equal')
        self.ax.set_title("Кривая Леви. Итерация: 0")
        self.line.set_data([], [])
        return self.line,

    def update_frame(self, frame):
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_title(f"Кривая Леви. Итерация: {frame}")
        x, y = self.iter_data[frame]
        self.ax.plot(x, y, 'b-', lw=1)
        return self.line,

    def animate(self):
        frames = self.max_iter + 1
        ani = FuncAnimation(self.fig, self.update_frame, frames=frames,
                            init_func=self.init_plot, interval=1000, blit=False, repeat=False)
        plt.show()


class FractalTreeAnimator:
    def __init__(self, max_iter=5, angle=np.pi/6, scale=0.7):
        """
        max_iter - глубина рекурсии
        angle - угол разветвления
        scale - коэффициент уменьшения длины веток
        Стартовая ветка: из (0,0) вверх длиной 1.
        Каждая ветка порождает две новых под углами +angle и -angle и длиной умноженной на scale.
        """
        self.max_iter = max_iter
        self.angle = angle
        self.scale = scale
        # Генерируем ветви для каждой итерации
        self.iter_branches = self.generate_iterations()

        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.lines = []

    def generate_iterations(self):
        # Каждая итерация - список веток, где ветка - кортеж ((x1,y1),(x2,y2))
        # Итерация 0: одна ветка (0,0) -> (0,1)
        iterations = []
        current = [((0,0),(0,1))]
        iterations.append(current)

        for i in range(1, self.max_iter+1):
            next_level = []
            for branch in current:
                (x1,y1),(x2,y2) = branch
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx*dx+dy*dy)

                # Конечная точка этой ветки - начало двух новых
                bx, by = x2, y2
                # Угол текущей ветки
                base_angle = np.arctan2(dy,dx)

                # Первая ветка
                angle_left = base_angle + self.angle
                x3 = bx + length*self.scale*np.cos(angle_left)
                y3 = by + length*self.scale*np.sin(angle_left)

                # Вторая ветка
                angle_right = base_angle - self.angle
                x4 = bx + length*self.scale*np.cos(angle_right)
                y4 = by + length*self.scale*np.sin(angle_right)

                next_level.append(((x2,y2),(x3,y3)))
                next_level.append(((x2,y2),(x4,y4)))
            iterations.append(next_level)
            current = current + next_level
        # Обратите внимание: итерация i включает все ветки до i-го уровня (кумулятивно)
        # Если нужно только текущий уровень, можно изменить логику.
        # Здесь мы делаем так, что на i-й итерации показываем все ветки, созданные до этого момента.
        return iterations

    def init_plot(self):
        self.ax.set_aspect('equal')
        self.ax.set_title("Фрактальное дерево. Итерация: 0")
        self.ax.set_xlim(-2,2)
        self.ax.set_ylim(0,2.5)
        return []

    def update_frame(self, frame):
        # frame = 0: начальная ветка
        # frame = 1: добавили новый уровень, и так далее.
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_title(f"Фрактальное дерево. Итерация: {frame}")
        self.ax.set_xlim(-2,2)
        self.ax.set_ylim(0,2.5)

        # Отрисовываем все ветки до текущей итерации включительно
        all_branches = []
        for i in range(frame+1):
            for branch in self.iter_branches[i]:
                all_branches.append(branch)

        for (x1,y1),(x2,y2) in all_branches:
            self.ax.plot([x1,x2],[y1,y2],'g-')

        return []

    def animate(self):
        frames = self.max_iter+1
        ani = FuncAnimation(self.fig, self.update_frame, frames=frames,
                            init_func=self.init_plot, interval=1000, blit=False, repeat=False)
        plt.show()

if __name__ == "__main__":
    print('Выберите фрактал: \n1) Треугольник Серпинского\n2) Множество Мандельброта\n3) Множество Жюлиа\n4) Кривая Коха\n'
          '5) Кривая Леви\n6) Дракон Хартера-Хейтуэя\n7) Дерево')
    choice = int(input())
    animator = None
    if choice == 1:
        max_it = get_param('Введите максимальное кол-во итераций', 5, int)
        animator = SierpinskiAnimator(max_iter=max_it)

    elif choice == 2:
        start_iter = get_param("Введите начальное число итераций", 10, int)
        end_iter = get_param("Введите конечное число итераций", 200, int)
        step = get_param("Введите шаг увеличения итераций", 10, int)
        resolution = get_param("Введите разрешение (например 300)", 300, int)
        zoom = get_param("Введите zoom (по умолчанию 1.0)", 1.0, float)
        cx = get_param("Введите x-координату центра (по умолчанию -0.5)", -0.5, float)
        cy = get_param("Введите y-координату центра (по умолчанию 0)", 0.0, float)
        animator = MandelbrotAnimator(start_iter=start_iter, end_iter=end_iter, step=step, resolution=resolution, zoom=zoom, center=(cx, cy))

    elif choice == 3:
        start_iter = get_param("Введите начальное число итераций", 10, int)
        end_iter = get_param("Введите конечное число итераций", 200, int)
        step = get_param("Введите шаг увеличения итераций", 10, int)
        resolution = get_param("Введите разрешение (например 300)", 300, int)
        zoom = get_param("Введите zoom (по умолчанию 1.0)", 1.0, float)
        c_real = get_param("Введите действительную часть c (по умолчанию -0.7)", -0.7, float)
        c_imag = get_param("Введите мнимую часть c (по умолчанию 0.27015)", 0.27015, float)
        animator = JuliaAnimator(c=(c_real, c_imag), start_iter=start_iter, end_iter=end_iter, step=step, resolution=resolution, zoom=zoom)

    elif choice == 4:
        max_iter = get_param("Введите максимальное число итераций для кривой Коха", 5, int)
        animator = KochAnimator(max_iter=max_iter)

    elif choice == 5:
        max_iter = get_param("Введите максимальное число итераций для кривой Леви", 5, int)
        angle = get_param('Введите угол', 45, int)
        animator = LeviAnimator(max_iter=max_iter, angle_degrees=angle)
    elif choice == 6:
        max_iter = get_param("Введите максимальное число итераций для дракона Хартера-Хэйтуэя", 5, int)
        animator = HeighwayDragonAnimator(max_iter=max_iter)
    elif choice == 7:
        max_iter = get_param("Введите максимальное число итераций для Дерева", 5, int)
        angle = get_param('Введите угол', np.pi/6, float)
        scale = get_param('Введите коэффициент уменьшения длины веток', 0.7, float)
        animator = FractalTreeAnimator()
    else:
        print('Вы ввели неверное значение')

    if animator is not None:
        animator.animate()
