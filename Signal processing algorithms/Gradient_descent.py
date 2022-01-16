import time
import numpy as np
import matplotlib.pyplot as plt

"""
Градиентный спуск при одном параметра параметрах: y = k * x + b. Ищем x
"""

# Функция


def f(x):
    return x * x - 5 * x + 5
# Ее производная


def df(x):
    return 2 * x - 5


N = 100  # number of iteration
xx = 0  # first value
lmd = 0.1  # step

x_plt = np.arange(0, 5.0, 0.1)
f_plt = [f(x) for x in x_plt]

# Пострение графика
plt.ion()  # включение интерактивного режима
fig, ax = plt.subplots()  # создание окна и осей
ax.grid(True)  # сетка

ax.plot(x_plt, f_plt)
point = ax.scatter(xx, f(xx), c='red')  # jnj,hf;tybt njxrb

# Алгоритм градиента
for i in range(N):
    xx = xx - lmd * df(xx)  # формула градиентного спуска

    point.set_offsets([xx, f(xx)])  # смещение точки

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)  # время обновления

plt.ioff()  # выключение интерактивного режима отображения графиков
print(xx)
ax.scatter(xx, f(xx), c='blue')
plt.show()
