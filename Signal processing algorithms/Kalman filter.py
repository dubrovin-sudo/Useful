import numpy as np
import matplotlib.pyplot as plt


""""""
N = 1000        # число наблюдений
dNoise = 1      # дисперсия шума
dSignal = 5     # диспресия сигнала
r = 0.99        # коэффициент корреляции в модели движения
en = 0.1        # диспресия СВ в модели движения (см. марковское уравнение)

# Моделирование марковской последовательности
mark = np.zeros(N)                         # истинные координаты перемещения (пока просто нули)
mark[0] = np.random.normal(0, dSignal)     # формирование первой координаты
for i in range(1, N):                   # формирование последующих координат по модели АР
    mark[i] = r * mark[i - 1] + np.random.normal(0, en) # уравнение следующего шага в марковском процессе

z = mark + np.random.normal(0, dNoise, N)  # формирование наблюдений

# фильтрация сигнала с помощью фильтра Калмана
xx = np.zeros(N)    # вектор для хранения оценок перемещений
P = np.zeros(N)     # вектор для хранения дисперсий ошибок оценивания
xx[0] = z[0]        # первая оценка (первое наблюдение сигнала)
P[0] = dNoise       # дисперсия первой оценки (нашего шума)

# реккурентное вычисление оценок по фильтру Калмана
for i in range(1, N):
    Pe = r * r * P[i - 1] + en * en         # ошибка прогнозирования
    P[i] = (Pe * dNoise) / (Pe + dNoise)    # ошибка прогнозирования для текущего местоположения
    xx[i] = r * xx[i - 1] + P[i] / dNoise * (z[i] - r * xx[i - 1])

# отображение результатов
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), sharex='col')
ax1.plot(mark)
ax1.plot(z)
# ax1.plot(xx)
ax1.set_title('Марковская последовательность+шум и ее фильтрация')

ax1.grid(True)

ax2.plot(P)
ax2.set_title('Ошибка прогнозирования текущего местаположения')
ax2.set_xlabel(f'{N} Steps')
ax2.grid(True)
plt.show()
