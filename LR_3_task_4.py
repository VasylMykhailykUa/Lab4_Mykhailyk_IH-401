# Імпортування необхідних інструментів для аналізу даних
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Завантаження вбудованого набору даних про діабет
diabetes_data = load_diabetes()
X_predictors = diabetes_data.data     # Матриця предикторів (10 змінних)
y_target = diabetes_data.target      # Цільова змінна (прогрес діабету)

# Ініціалізація та навчання моделі регресії
regression_model = LinearRegression()
regression_model.fit(X_predictors, y_target)

# Прогнозування значень
y_predicted = regression_model.predict(X_predictors)

# Оцінка якості моделі
r2_accuracy = r2_score(y_target, y_predicted)
mae_error = mean_absolute_error(y_target, y_predicted)
mse_error = mean_squared_error(y_target, y_predicted)

# Створення графіку порівняння реальних та прогнозованих значень
plt.figure(figsize=(10, 6))
plt.scatter(y_target, y_predicted, color='blue', alpha=0.5)
plt.plot([y_target.min(), y_target.max()], [y_target.min(), y_target.max()], 'r--', lw=2)
plt.xlabel('Реальні значення')
plt.ylabel('Прогнозовані значення')
plt.title('Порівняння реальних та прогнозованих значень')
plt.grid(True)
plt.show()

# Виведення результатів аналізу
print("\nРезультати аналізу множинної регресії:")
print(f"Коефіцієнти впливу змінних: {regression_model.coef_}")
print(f"Константа (перетин з віссю Y): {regression_model.intercept_:.3f}")
print("\nМетрики якості моделі:")
print(f"R² (коефіцієнт детермінації): {r2_accuracy:.3f}")
print(f"MAE (середня абсолютна помилка): {mae_error:.3f}")
print(f"MSE (середньоквадратична помилка): {mse_error:.3f}")

