import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Градиентный спуск при двух параметрах: y = k * x + b. Ищем k и b
"""

# Критерий ошибки - сумма квадратов ошибок отклонений, ее надо уменьшить
# E = (y-kx-b)^2
def E(y, k, b):
    ff = np.array([k * x + b for x in range(N)])
    return np.dot((y - ff).T, (y - ff))

# Градиент по k
def dEdk(y, k, b):
    ff = np.array([k * x + b for x in range(N)])
    return -2 * np.dot((y - ff).T, range(N))

# Градиент по b
def dEdb(y, k, b):
    ff = np.array([k * x + b for x in range(N)])
    return -2 * (y - ff).sum()


N = 100  # число точек
Niter = 50  # число итераций
sigma = 3  # стандартное отклонение наблюдаемых значений
k_theory = 0.5  # теоретическое значение параметра k
b_theory = 2  # теоретическое значение параметра а

# Начальные значение
k_begin = 0
b_begin = 0
lmd1 = 0.000001
lmd2 = 0.0005

# Теоретическая прямая
f = np.array([k_theory * x + b_theory for x in range(N)])
# Экспериментальные значения y
y = np.array(f + np.random.normal(0, sigma, N))

# Создание поверхности
k_plt = np.arange(-1, 2, 0.1)
b_plt = np.arange(0, 3, 0.1)
E_plt = np.array([[E(y, k, b) for k in k_plt]
                 for b in b_plt])  # функция ошибок для визуализации


# Построение графика
plt.ion()  # включение интерактивного режима
fig = plt.figure()
ax = Axes3D(fig)  # формирование трехмерной оси

k, b = np.meshgrid(k_plt, b_plt)
ax.plot_surface(k, b, E_plt, color='y', alpha=0.5)

ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('E')


point = ax.scatter(k_begin, b_begin, E(
    y, k_begin, b_begin), color='red')  # точка

# Алгоритм градиента
for i in range(Niter):
    # формула градиентного спуска для k
    k_begin = k_begin - lmd1 * dEdk(y, k_begin, b_begin)
    # формула градиентного спуска для b
    b_begin = b_begin - lmd2 * dEdb(y, k_begin, b_begin)

    point = ax.scatter(
        k_begin, b_begin, E(
            y, k_begin, b_begin), color='red')  # смещение точки

    # перерисовка графика и задержка 10мс
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)  # время обновления

    print(k_begin, b_begin)

plt.ioff()  # выключение интерактивного режима отображения графиков
plt.show()

#   Аппроксимированная прямая
ff = np.array([k_begin * x + b_begin for x in range(N)])
plt.scatter(range(N), y, s=2, color='red')
plt.plot(f)  # теоретическая прямая
plt.plot(ff, c='red')  # аппроксимированная прямая
plt.grid(True)
plt.show()
