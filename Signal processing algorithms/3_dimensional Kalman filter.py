import numpy as np
import matplotlib.pyplot as plt


""""""
N = 1000        # число наблюдений
dNoise = 1      # дисперсия шума
dSignal = 5     # диспресия сигнала
r = 0.99        # коэффициент корреляции в модели движения
en = 0.1        # диспресия СВ в модели движения (см. марковское уравнение)

M = 3                                               # размерность вектора координат положения объекта
R = np.array([[r, 0, 0], [0, r, 0], [0, 0, r]])     # диагональная матрица с дисперсиями en по главное диагонали
Vksi = np.eye(M) * en                               # диагональная матрица с дисперсиями ошибок наблюдений
V = np.eye(M) * dNoise

# Моделирование марковской последовательности
mark = np.zeros(N * M).reshape(N, M)                # истинные координаты перемещения (пока просто нули)
mark[:][0] = np.random.normal(0, dSignal)           # формирование первой координаты
for i in range(
        1, N):                                      # формирование последующих координат по модели АР
    # уравнение следующего шага в марковском процессе
    mark[:][i] = np.dot(R, mark[:][i - 1]) + np.random.normal(0, en, M)

z = mark + np.random.normal(0, dNoise, size=(N, M))  # формирование наблюдений

# фильтрация сигнала с помощью фильтра Калмана
xx = np.zeros(N * M).reshape(N, M)                  # вектор для хранения оценок перемещений
P = np.zeros(N * M).reshape(N, M)                   # вектор для хранения дисперсий ошибок оценивания
xx[:][0] = z[:][0]                                  # первая оценка (первое наблюдение сигнала)
P = V                                               # дисперсия первой оценки (нашего шума)


Vinv = np.linalg.inv(V)                             # вычисление обратной матрицы дисперсии ошибок наблюдений

# реккурентное вычисление оценок по фильтру Калмана
for i in range(1, N):
    Pe = np.dot(np.dot(R, P), R.T) + Vksi           # ошибка прогнозирования
    P = np.dot(Pe, V) * np.linalg.inv(Pe + V)       # ошибка прогнозирования для текущего местоположения
    xe = np.dot(R, xx[:][i - 1])

# отображение результатов
fig, (axX, axY, axZ) = plt.subplots(
    nrows=3, ncols=1, figsize=(10, 6), sharex='col')

res = xx.reshape(M * N)
resX = mark.reshape(M * N)
resZ = z.reshape(M * N)

axX.plot(resX[0:N * M:M])
axX.plot(resZ[0:N * M:M])
axX.plot(res[0:N * M:M])
axY.plot(resX[0:N * M:M])
axY.plot(resZ[0:N * M:M])
axY.plot(res[0:N * M:M])
axZ.plot(resX[0:N * M:M])
axZ.plot(resZ[0:N * M:M])
axZ.plot(res[0:N * M:M])


axX.set_ylabel('Ось X')
axY.set_ylabel('Ось Y')
axZ.set_ylabel('Ось Z')

axX.grid(True)
axY.grid(True)
axZ.grid(True)
plt.show()
