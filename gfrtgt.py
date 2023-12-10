
from math import *
import random as rnd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import minimize

import imageio
import os
import time


def Ackley(xy):
    x, y = xy
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20




initial_guess = [-2, 2]

final_values = []


start_time = time.time()
for _ in range(100):
    result = minimize(Ackley, initial_guess, method='BFGS')
    final_values.append(result.fun)
end_time = time.time()




print("Математическое ожидание финального значения функции Ackley:", np.mean(final_values))
print("Дисперсия финального значения функции Ackley:", np.var(final_values))
print("Минимум найден в точке:", result.x)
print("Минимальное значение функции Ackley:", result.fun)
print("Время выполнения:", end_time - start_time)




# Создаем сетку значений
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = Ackley([X, Y])  # Теперь передаем один вектор переменных [X, Y]

# Построение 3D-графика
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Настройка меток и заголовка
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Ackley Function Value')
ax.set_title('3D Plot of Ackley Function')

# Отображение графика
plt.show()

# Контурный график
plt.contour(X, Y, Z, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot of Ackley Function')
plt.show()






def Ackley(x, y):
    return -20*exp(-0.2*sqrt(0.5*(x**2+y**2)))-exp(0.5*(cos(2*pi*x)+cos(2*pi*y)))+e+20



class Unit:

    def __init__(self, start, end, currentVelocityRatio, localVelocityRatio, globalVelocityRatio, function):
        self.start = start
        self.end = end
        
        self.currentVelocityRatio = currentVelocityRatio
        self.localVelocityRatio = localVelocityRatio
        self.globalVelocityRatio = globalVelocityRatio
        
        
        self.function = function
        
        
        self.localBestPos = self.getFirsPos()
        self.localBestScore = self.function(*self.localBestPos)
        
        
        self.currentPos = self.localBestPos[:]
        self.score = self.function(*self.localBestPos)
        
        
        self.globalBestPos = []
        
        self.velocity = self.getFirstVelocity()


    def getFirstVelocity(self):
        """ Метод для задания первоначальной скорости"""
        minval = -(self.end - self.start)
        maxval = self.end - self.start
        return [rnd.uniform(minval, maxval), rnd.uniform(minval, maxval)]

    def getFirsPos(self):
        """ Метод для получения начальной позиции"""
        return [rnd.uniform(self.start, self.end), rnd.uniform(self.start, self.end)]


    def nextIteration(self):
        """ Метод для нахождения новой позиции частицы"""
        rndCurrentBestPosition = [rnd.random(), rnd.random()]
        rndGlobalBestPosition = [rnd.random(), rnd.random()]
        # делаем перерасчет скорости частицы исходя из всех введенных параметров
        velocityRatio = self.localVelocityRatio + self.globalVelocityRatio
        commonVelocityRatio = 2 * self.currentVelocityRatio / abs(2-velocityRatio-sqrt(velocityRatio ** 2 - 4 * velocityRatio))
        multLocal = list(map(lambda x: x*commonVelocityRatio * self.localVelocityRatio, rndCurrentBestPosition))
        betweenLocalAndCurPos = [self.localBestPos[0] - self.currentPos[0], self.localBestPos[1] - self.currentPos[1]]
        betweenGlobalAndCurPos = [self.globalBestPos[0] - self.currentPos[0], self.globalBestPos[1] - self.currentPos[1]]
        multGlobal = list(map(lambda x: x*commonVelocityRatio * self.globalVelocityRatio, rndGlobalBestPosition))
        newVelocity1 = list(map(lambda coord: coord * commonVelocityRatio, self.velocity))
        newVelocity2 = [coord1 * coord2 for coord1, coord2 in zip(multLocal, betweenLocalAndCurPos)]
        newVelocity3 = [coord1 * coord2 for coord1, coord2 in zip(multGlobal, betweenGlobalAndCurPos)]
        self.velocity = [coord1 + coord2 + coord3 for coord1, coord2, coord3 in zip(newVelocity1, newVelocity2, newVelocity3)]
        # передвигаем частицу и смотрим, какое значение целевой фунции получается
        self.currentPos = [coord1 + coord2 for coord1, coord2 in zip(self.currentPos, self.velocity)]
        newScore = self.function(*self.currentPos)
        if newScore < self.localBestScore:
            self.localBestPos = self.currentPos[:]
            self.localBestScore = newScore
        return newScore


class Swarm:

    def __init__(self, sizeSwarm,
                 currentVelocityRatio,
                 localVelocityRatio,
                 globalVelocityRatio,
                 numbersOfLife,
                 function,
                 start,
                 end):
        self.sizeSwarm = sizeSwarm
        
        self.currentVelocityRatio = currentVelocityRatio
        self.localVelocityRatio = localVelocityRatio
        self.globalVelocityRatio = globalVelocityRatio


        self.numbersOfLife = numbersOfLife
        
        self.function = function
        
        self.start = start
        self.end = end
        
        self.swarm = []
        
        self.globalBestPos = []
        self.globalBestScore = float('inf')

        self.createSwarm()


    def createSwarm(self):
        """ Метод для создания нового роя"""
        pack = [self.start, self.end, self.currentVelocityRatio, self.localVelocityRatio, self.globalVelocityRatio, self.function]
        self.swarm = [Unit(*pack) for _ in range(self.sizeSwarm)]
        for unit in self.swarm:
            if unit.localBestScore < self.globalBestScore:
                self.globalBestScore = unit.localBestScore
                self.globalBestPos = unit.localBestPos



    def startSwarm(self):
        """ Метод для запуска алгоритма"""
        dataForGIF = []
        local_extremum_times = [] 
        final_start = time.time()
        for _ in range(self.numbersOfLife):
            oneDataX = []
            oneDataY = []
            for unit in self.swarm:
                oneDataX.append(unit.currentPos[0])
                oneDataY.append(unit.currentPos[1])
                unit.globalBestPos = self.globalBestPos
                score = unit.nextIteration()
                if score < self.globalBestScore:
                    self.globalBestScore = score
                    self.globalBestPos = unit.localBestPos
                    end_time = time.time()
                    local_extremum_times.append(end_time - final_start)  
            dataForGIF.append([oneDataX, oneDataY])
        final_end = time.time()
        mean_time = np.mean(dataForGIF)
        variance_time = np.var(dataForGIF)

        print("Математическое ожидание времени нахождения локального экстремума:", mean_time)
        print("Дисперсия времени нахождения локального экстремума:", variance_time)
        print("Время по окончанию итераций", final_end-final_start)
        print("Время нахождения последнего экстремума", end_time-final_start)

        
#         # рисуем gif
#         fnames = []
#         i = 0
#         # dataForGIF = []
#         for x, y in dataForGIF:
#             i += 1
#             fname = f"g{i}.png"
#             fig, (ax1, ax2) = plt.subplots(1, 2)
#             fig.suptitle(f"Итерация: {i}")
#             ax2.plot(x, y, 'bo')
#             ax2.set_xlim(self.start, self.end)
#             ax2.set_ylim(self.start, self.end)
#             ax1.plot(x, y, 'bo')
#             fig.savefig(fname)
#             plt.close()
#             fnames.append(fname)

#         with imageio.get_writer('swarm.gif', mode='I') as writer:
#             for filename in fnames:
#                 image = imageio.imread(filename)
#                 writer.append_data(image)

#         for filename in set(fnames):
#             os.remove(filename)



a = Swarm(650, 0.1, 1, 5, 100, Ackley, -5, 5)
a.startSwarm()
print("РЕЗУЛЬТАТ:", a.globalBestScore, "В ТОЧКЕ:",a.globalBestPos)
