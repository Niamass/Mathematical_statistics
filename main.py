import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import math as m

size = [10, 50, 1000]

#Нормальное распределение
for s in size:
    density = sts.norminvgauss(1, 0)  # плотность
    hist_sample = sts.norminvgauss.rvs(1, 0, size=s)  # выборка для гистограммы

    fig, ax = plt.subplots()
    ax.hist(hist_sample, density=True)  # гистограмма
    x = np.linspace(density.ppf(0.01), density.ppf(0.99), 100)
    ax.plot(x, density.pdf(x), 'k-', lw=2)  # плотность
    ax.set_xlabel("Normal")
    ax.set_ylabel("Density")
    ax.set_title("Sample size = " + str(s))
    plt.show()


#Распределение Коши
for s in size:
    density = sts.cauchy()  # плотность
    hist_sample = sts.cauchy.rvs(size=s)  # выборка для гистограммы

    fig, ax = plt.subplots()
    ax.hist(hist_sample, density=True)  # гистограмма
    x = np.linspace(density.ppf(0.01), density.ppf(0.99), 100)
    ax.plot(x, density.pdf(x), 'k-', lw=2)  # плотность
    ax.set_xlabel("Cauchy")
    ax.set_ylabel("Density")
    ax.set_title("Sample size = " + str(s))
    plt.show()


#Распределение Лапласа
for s in size:
    density = sts.laplace(loc=0, scale=1 / m.sqrt(2))  # плотность
    hist_sample = sts.laplace.rvs(loc=0, scale=1 / m.sqrt(2), size=s)  # выборка для гистограммы

    fig, ax = plt.subplots()
    ax.hist(hist_sample, density=True)  # гистограмма
    x = np.linspace(density.ppf(0.01), density.ppf(0.99), 100)
    ax.plot(x, density.pdf(x), 'k-', lw=2)  # плотность
    ax.set_xlabel("Laplace")
    ax.set_ylabel("Density")
    ax.set_title("Sample size =" + str(s))
    plt.show()


#Распределение Пуассона
for s in size:
    density = sts.poisson(mu=10)  # плотность
    hist_sample = sts.poisson.rvs(mu=10, size=s)  # выборка для гистограммы

    fig, ax = plt.subplots()
    ax.hist(hist_sample, density=True)  # гистограмма
    x = np.arange(sts.poisson.ppf(0.01, mu=10), sts.poisson.ppf(0.99, mu=10))
    ax.plot(x, density.pmf(x), 'k-', lw=2)  # плотность
    ax.set_xlabel("Poisson")
    ax.set_ylabel("Density")
    ax.set_title("Sample size = " + str(s))
    plt.show()

#Равномерное распределение
for s in size:
    density = sts.uniform(-m.sqrt(3), 2 * m.sqrt(3))  # плотность
    hist_sample = sts.uniform.rvs(-m.sqrt(3), 2 * m.sqrt(3), size=s)  # выборка для гистограммы

    fig, ax = plt.subplots()
    ax.hist(hist_sample, density=True)  # гистограмма
    x = np.linspace(density.ppf(0.01), density.ppf(0.99), 100)
    ax.plot(x, density.pdf(x), 'k-', lw=2)  # плотность
    ax.set_xlabel("Uniform")
    ax.set_ylabel("Density")
    ax.set_title("Sample size = " + str(s))
    plt.show()




