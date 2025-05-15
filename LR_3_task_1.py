import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Завантаження даних
data = np.loadtxt('attached_assets/data_singlevar_regr.txt', delimiter=',')
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
print("Mean absolute error =", round(mean_absolute_error(y, y_pred), 2))
print("Mean squared error =", round(mean_squared_error(y, y_pred), 2))
print("Median absolute error =", round(median_absolute_error(y, y_pred), 2))
print("Explain variance score =", round(explained_variance_score(y, y_pred), 2))
print("R2 score =", round(r2_score(y, y_pred), 2))

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
