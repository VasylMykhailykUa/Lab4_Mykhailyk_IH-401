import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Генерація даних з синусоїдальною залежністю
m = 100
X = np.linspace(-3, 3, m)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, m)

# Підготовка даних для sklearn
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Лінійна регресія
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

# Поліноміальна регресія
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

# Оцінка якості моделей
linear_mse = mean_squared_error(y, y_linear_pred)
poly_mse = mean_squared_error(y, y_poly_pred)
linear_r2 = r2_score(y, y_linear_pred)
poly_r2 = r2_score(y, y_poly_pred)

# Виведення результатів
print("\nОцінка якості моделей:")
print("Лінійна регресія:")
print(f"MSE: {linear_mse:.3f} | R²: {linear_r2:.3f}")
print("Поліноміальна регресія:")
print(f"MSE: {poly_mse:.3f} | R²: {poly_r2:.3f}")

# Візуалізація результатів
plt.figure(figsize=(15, 6))

# Графік лінійної регресії
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', alpha=0.5, label='Дані')
plt.plot(X, y_linear_pred, color='red', label='Лінійна модель')
plt.title('Лінійна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Графік поліноміальної регресії
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', alpha=0.5, label='Дані')
sorted_idx = np.argsort(X.flatten())
plt.plot(X[sorted_idx], y_poly_pred[sorted_idx], color='green', label='Поліноміальна модель')
plt.title('Поліноміальна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('regression_analysis.png')
plt.show()
