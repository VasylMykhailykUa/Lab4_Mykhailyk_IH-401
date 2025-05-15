import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Завантаження даних
data = np.loadtxt('attached_assets/data_regr_2.txt', delimiter=',')
X = data[:, 0].reshape(-1, 1)  # Матриця ознак
y = data[:, 1]                 # Вектор цільових значень

# Створення та тренування моделі
regressor = LinearRegression()
regressor.fit(X, y)

# Прогнозування
y_pred = regressor.predict(X)

# Розрахунок метрик
from sklearn.metrics import mean_absolute_error, median_absolute_error, explained_variance_score

print("Linear regressor performance:")
print("Mean absolute error = {:.3f}".format(mean_absolute_error(y, y_pred)))
print("Mean squared error = {:.3f}".format(mean_squared_error(y, y_pred)))
print("Median absolute error = {:.3f}".format(median_absolute_error(y, y_pred)))
print("Explain variance score = {:.3f}".format(explained_variance_score(y, y_pred)))
print("R2 score = {:.3f}".format(r2_score(y, y_pred)))

# Візуалізація результатів
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='green', alpha=0.6, label='Точки даних')
plt.plot(X, y_pred, color='purple', linewidth=2, label='Регресійна пряма')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Лінійна регресія')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('regression_plot.png')
plt.show()
