import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Зчитування даних із файлу для аналізу впливу трьох незалежних змінних
# на цільову змінну (множинна регресія)
raw_data = np.loadtxt('attached_assets/data_multivar_regr.txt', delimiter=',')

# Розділення даних на матрицю предикторів (X) та вектор відгуків (y)
X_features = raw_data[:, :-1]  # Перші три стовпці - незалежні змінні
y_target = raw_data[:, -1]    # Останній стовпець - залежна змінна

# Ініціалізація та навчання моделі множинної регресії
predictor = LinearRegression()
predictor.fit(X_features, y_target)

# Прогнозування значень на основі навчених коефіцієнтів
y_predicted = predictor.predict(X_features)

# Обчислення метрик якості моделі
mse_score = mean_squared_error(y_target, y_predicted)
r2_score_val = r2_score(y_target, y_predicted)

# Виведення результатів аналізу
print("\nАналіз множинної регресії:")
print(f"Коефіцієнти впливу змінних: {predictor.coef_}")
print(f"Вільний член рівняння: {predictor.intercept_:.3f}")
print(f"\nЯкість моделі:")
print(f"Середньоквадратична помилка: {mse_score:.3f}")
print(f"Коефіцієнт детермінації R²: {r2_score_val:.3f}")
