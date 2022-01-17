import numpy as np
import matplotlib.pyplot as plt



"""Пример марсковской последовательности сформированной на основе шума с дисперсией 1"""
N = 100000
sigma = 1

fSignal = np.zeros(N)
fNoise = np.random.normal(0, sigma, N)
for i in range(1, N):
    fSignal[i] = fSignal[i-1] + fNoise[i]
    # дисперсия накапливается

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey='row')
ax1.plot(fNoise)
ax1.plot(fSignal)
ax1.set_title('Марковская последовательность')
ax1.grid(True)

"""Пример марсковской последовательности сформированной на основе шума с постоянной дисперсией en"""

N = 1000
sigma = 1
r = 0.99
en = np.sqrt((1-r*r)*sigma*sigma) # авторегрессия первого порядка

fSignal = np.zeros(N)
fSignal[0] = np.random.normal(0, sigma) # формируем первый отсчет сигнала

for i in range(1, N):
    fSignal[i] = r*fSignal[i-1] + np.random.normal(0, en)

ax2.plot(fSignal)
ax2.set_title(f'Марковская последовательность c дисперсией en = {en}')
ax2.grid(True)
plt.show()
