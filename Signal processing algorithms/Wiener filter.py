import numpy as np
import matplotlib.pyplot as plt


""""""
N = 30              # число наблюдений
dNoise = 0.01       # дисперсия шума
en = 0.01           # диспресия сигнала (для моделирования истинного веча человека)
mark0 = 76             # начальная масса
r = 0.9             # коэффициент корреляции в модели движения (см. марковское уравнение)
en = 0.1            # диспресия СВ в модели движения (см. марковское уравнение)
ex = en/(1-r*r)     # диспресия сигнала относительно 0

# Моделирование марковской последовательности (для простоты)
mark = np.zeros(N)          # истинные координаты перемещения (пока просто нули)
mark[0] = 0                 # формирование первой координаты
for i in range(1, N):       # формирование последующих координат по модели АР
    mark[i] = r * mark[i - 1] + np.random.normal(0, en) # уравнение следующего шага в марковском процессе

mark +=mark0

z = mark + np.random.normal(0, dNoise, N)  # формирование наблюдений

# формируем вспомогательные матрицы
R = np.array([[r**np.abs(i-j) for j in range(N)] for i in range(N)])    # катрица корреляции между отсчетами СВ
V = np.eye(N)*dNoise    # диагональная матрица для дисперсии шума наблюдений
Rvinv = np.linalg.inv(R+V/ex)        # обратная матрица

# построение оценок
mz = z.mean() # среднее значение наблюдений
xx = np.zeros(N)
for k in range(N):
    alfa = np.dot(R[:, k], Rvinv)
    xx[k] = np.dot(alfa, (z-mz)) + mz

# отображение результатов
fig, ax = plt.subplots()
ax.plot(mark)
ax.plot(z)
ax.plot(xx)
ax.set_title('Марковская последовательность+шум и ее фильтрация')

ax.grid(True)

plt.show()