import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import pandas
from sklearn.preprocessing import StandardScaler

# Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv.
# Целевая переменная записана в первом столбце, признаки — во втором и третьем.
test_data = pandas.read_csv('perceptron-test.csv', header=None)
train_data = pandas.read_csv('perceptron-train.csv', header=None)

X_test = test_data[[1, 2]]
y_test = test_data[0]
X_train = train_data[[1, 2]]
y_train = train_data[0]

# Обучите персептрон со стандартными параметрами и random_state=241.
clf = Perceptron(random_state=241, max_iter=5, tol=None)
clf.fit(X_train, y_train)

# прогоним тестовую выборку
predictions = clf.predict(X_test)

# # Подсчитайте качество (долю правильно классифицированных объектов, accuracy)
# # полученного классификатора на тестовой выборке.
# # В качестве метрики качества мы будем использовать долю верных ответов (accuracy).
# # Для ее подсчета можно воспользоваться функцией sklearn.metrics.accuracy_score,
# # первым аргументом которой является вектор правильных ответов, а вторым — вектор ответов алгоритма.
accuracy = accuracy_score(y_test, predictions)

# Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучите персептрон на новой выборке. Найдите долю правильных ответов на тестовой выборке.
clf = Perceptron(random_state=241, max_iter=5, tol=None)
clf.fit(X_train_scaled, y_train)
predictions_2 = clf.predict(X_test_scaled)
accuracy_2 = accuracy_score(y_test, predictions_2)

# Найдите разность между качеством на тестовой выборке после нормализации и качеством до нее.
# Это число и будет ответом на задание.
print(accuracy, accuracy_2)
print(round((accuracy_2 - accuracy), 3))
