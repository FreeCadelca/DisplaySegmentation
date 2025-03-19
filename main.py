import numpy as np
import scipy.stats as stats

# Данные
weights = np.array([812.0, 786.7, 794.1, 791.6, 811.1, 797.4, 797.8, 800.8, 793.2])
n = len(weights)  # Размер выборки
df = n - 1  # Степени свободы

# Выборочное стандартное отклонение
s = np.std(weights, ddof=1)
print(s)
# Квантили хи-квадрат распределения для 90% доверительного интервала
alpha = 0.10
chi2_lower = stats.chi2.ppf(alpha / 2, df)
chi2_upper = stats.chi2.ppf(1 - alpha / 2, df)

# Доверительный интервал для σ
sigma_lower = np.sqrt((df * s**2) / chi2_upper)
sigma_upper = np.sqrt((df * s**2) / chi2_lower)

print(sigma_lower, sigma_upper)

print(chi2_lower, chi2_upper)