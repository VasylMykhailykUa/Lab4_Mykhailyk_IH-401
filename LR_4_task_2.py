import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Вхідні експериментальні дані для аналізу
X = np.array([-1, -1, 0, 1, 2, 3]).reshape(-1, 1)  # Незалежна змінна
Y = np.array([-1, 0, 1, 1, 3, 5])  # Залежна змінна

# Ініціалізація та тренування моделі методом найменших квадратів
model = LinearRegression()
model.fit(X, Y)

# Визначення коефіцієнтів лінійної апроксимації
a = model.coef_[0]
b = model.intercept_
print(f"Апроксимуюча функція: Y = {a:.2f}X + {b:.2f}")

# Прогнозування Y для побудови прямої
Y_pred = model.predict(X)

# Побудова графіка
plt.scatter(X, Y, color='red', label='Експериментальні точки')
plt.plot(X, Y_pred, color='blue', label=f'Апроксимуюча пряма: Y = {a:.2f}X + {b:.2f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Апроксимація методом найменших квадратів')
plt.legend()
plt.grid(True)
plt.show()
