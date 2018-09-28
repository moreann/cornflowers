import sklearn
from sklearn.datasets import load_boston
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
import pandas
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

# Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston().
# Результатом вызова данной функции является объект,
# у которого признаки записаны в поле data, а целевой вектор — в поле target.
all_data = load_boston()

name = all_data['feature_names']
data = all_data['data']
target = all_data['target']
# Приведите признаки в выборке к одному масштабу при помощи функции sklearn.preprocessing.scale.
X = scale(data)
y = target
# target_s = sklearn.preprocessing.scale(target)

# Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом,
# чтобы всего было протестировано 200 вариантов (используйте функцию numpy.linspace).

a = np.linspace(1, 10, 200)
# Используйте KNeighborsRegressor с n_neighbors=5 и weights='distance' — данный параметр добавляет в алгоритм веса,
# зависящие от расстояния до ближайших соседей. В качестве метрики качества используйте среднеквадратичную ошибку
# (параметр scoring='mean_squared_error' у cross_val_score;
# при использовании библиотеки scikit-learn версии 0.18.1 и выше необходимо указывать scoring='neg_mean_squared_error').
#  Качество оценивайте, как и в предыдущем задании, с помощью кросс-валидации по 5 блокам с random_state = 42,
# не забудьте включить перемешивание выборки (shuffle=True).
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores2 = list()
for p in a:
    model = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
    scores2.append(cross_val_score(model,
                                   X,
                                   y,
                                   cv=kf,
                                   scoring='neg_mean_squared_error'))

# Определите, при каком p качество на кросс-валидации оказалось оптимальным.
# Обратите внимание, что cross_val_score возвращает массив показателей качества по блокам;
# необходимо максимизировать среднее этих показателей. Это значение параметра и будет ответом на задачу.
res2 = pandas.DataFrame(scores2, a).mean(axis=1).sort_values(ascending=False)

# print(res2)
top = res2.head(5)
print(top)
# 1.000000   -16.050209
# 1.090452   -16.367229
# 1.045226   -16.404081
# 1.135678   -16.442539
# 1.180905   -16.455281

# почему-то правильный ответ 1.18
