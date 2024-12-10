import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

matplotlib.use('TkAgg')

# Начальные параметры
angle_index = 1  # Начальный индекс угла
angle_options = [4 * np.pi, 6 * np.pi, 8 * np.pi]  # 2, 3 или 4 оборота
angle_labels = ['2 оборота', '3 оборота', '4 оборота']

# Генерация данных для спирали
def generate_spiral(angle):
    num_points = 1000
    theta = np.linspace(0, angle, num_points)
    r = theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

x, y = generate_spiral(angle_options[angle_index])

# Настройка фигуры и осей
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # Оставляем место внизу для кнопок

line, = ax.plot([], [], lw=2)
ax.set_xlim(-angle_options[2], angle_options[2])  # Максимальные пределы для всех углов
ax.set_ylim(-angle_options[2], angle_options[2])
ax.set_title('Анимация спирали')

# Добавление кнопок
button_axes = []
buttons = []
for i in range(3):
    ax_button = plt.axes([0.1 + i*0.3, 0.05, 0.2, 0.075])  # [left, bottom, width, height]
    button = Button(ax_button, angle_labels[i])
    button_axes.append(ax_button)
    buttons.append(button)

# Функции-обработчики для кнопок
def update_angle(index):
    global x, y, angle
    angle = angle_options[index]
    x, y = generate_spiral(angle)
    line.set_data([], [])
    ax.set_xlim(-angle_options[2], angle_options[2])
    ax.set_ylim(-angle_options[2], angle_options[2])
    ani.event_source.stop()  # Останавливаем текущую анимацию
    ani.frame_seq = ani.new_frame_seq()  # Сбрасываем последовательность кадров
    ani.event_source.start()  # Запускаем анимацию заново

def on_button_0(event):
    update_angle(0)

def on_button_1(event):
    update_angle(1)

def on_button_2(event):
    update_angle(2)

# Привязка кнопок к функциям
buttons[0].on_clicked(on_button_0)
buttons[1].on_clicked(on_button_1)
buttons[2].on_clicked(on_button_2)

# Функция инициализации
def init():
    line.set_data([], [])
    return line,

# Анимационная функция
def animate(i):
    if i < len(x):
        line.set_data(x[:i], y[:i])
    return line,

# Создание анимации
ani = animation.FuncAnimation(fig, animate, frames=len(x), init_func=init, interval=10, blit=True)

plt.show()
