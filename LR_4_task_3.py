import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# Створюємо експериментальні точки для аналізу
experimental_points_x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
experimental_points_y = np.array([3.2, 3.0, 1.0, 1.8, 1.9])

# Розрахунок коефіцієнтів за допомогою методу найменших квадратів
polynomial_coefficients = np.polyfit(experimental_points_x, experimental_points_y, deg=4)
interpolation_polynomial = Polynomial(polynomial_coefficients[::-1])

print("Знайдені коефіцієнти полінома:")
for index, coef in enumerate(polynomial_coefficients):
    print(f"Коефіцієнт при x^{4-index} = {coef:.4f}")

# Функція для обчислення значень полінома
def calculate_polynomial_value(x):
    return interpolation_polynomial(x)

# Візуалізація результатів
x_continuous = np.linspace(0, 0.8, 100)
y_continuous = calculate_polynomial_value(x_continuous)

plt.figure(figsize=(12, 8))
plt.scatter(experimental_points_x, experimental_points_y, 
           color='blue', s=100, label='Експериментальні точки', zorder=5)
plt.plot(x_continuous, y_continuous, 'r-', 
         label='Інтерполяційна крива', linewidth=2)
plt.xlabel('Значення x')
plt.ylabel('Значення y')
plt.title('Інтерполяція експериментальних даних')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.show()

# Перевірка точності інтерполяції
test_points = np.array([0.2, 0.5])
interpolated_values = calculate_polynomial_value(test_points)

print("\nРезультати обчислень у контрольних точках:")
for x, y in zip(test_points, interpolated_values):
    print(f"При x = {x}: f(x) = {y:.4f}")
