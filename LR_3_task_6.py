import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Генерація даних за варіантом 7
m = 100 
X = np.linspace(-3, 3, m).reshape(-1, 1)  # перетворення в стовпчик
y = np.sin(X) + np.random.uniform(-0.5, 0.5, (m, 1))  # додавання шуму

# Розділення на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Підготовка для поліноміальної регресії (наприклад, степінь 5 для синусоїди)
poly_features = PolynomialFeatures(degree=5, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Помилки
linear_train_errors = []
linear_test_errors = []

poly_train_errors = []
poly_test_errors = []

# Різні розміри тренувальних наборів
train_sizes = np.linspace(10, len(X_train), 20, dtype=int)

for size in train_sizes:
    # Лінійна модель
    linear_model = LinearRegression()
    linear_model.fit(X_train[:size], y_train[:size])
    y_train_pred = linear_model.predict(X_train[:size])
    y_test_pred = linear_model.predict(X_test)
    linear_train_errors.append(mean_squared_error(y_train[:size], y_train_pred))
    linear_test_errors.append(mean_squared_error(y_test, y_test_pred))
    
    # Поліноміальна модель
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly[:size], y_train[:size])
    y_train_poly_pred = poly_model.predict(X_train_poly[:size])
    y_test_poly_pred = poly_model.predict(X_test_poly)
    poly_train_errors.append(mean_squared_error(y_train[:size], y_train_poly_pred))
    poly_test_errors.append(mean_squared_error(y_test, y_test_poly_pred))

# Побудова кривих навчання
plt.figure(figsize=(12, 5))

# Лінійна регресія
plt.subplot(1, 2, 1)
plt.plot(train_sizes, linear_train_errors, 'r-+', label='Тренувальна помилка')
plt.plot(train_sizes, linear_test_errors, 'b-', label='Тестова помилка')
plt.title('Криві навчання (Лінійна регресія)')
plt.xlabel('Розмір тренувального набору')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

# Поліноміальна регресія
plt.subplot(1, 2, 2)
plt.plot(train_sizes, poly_train_errors, 'r-+', label='Тренувальна помилка')
plt.plot(train_sizes, poly_test_errors, 'b-', label='Тестова помилка')
plt.title('Криві навчання (Поліноміальна регресія, степінь 5)')
plt.xlabel('Розмір тренувального набору')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
