import pandas
import sklearn
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

data = pandas.read_csv('../../wine.data', names=['type', 'Alcohol','Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                                           'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                                           'Color intensity', 'Hue', 'diluted wines', 'Proline'])
# Технически кросс-валидация проводится в два этапа:
#
# Создается генератор разбиений sklearn.model_selection.KFold, который задает набор разбиений на обучение и валидацию.
# Число блоков в кросс-валидации определяется параметром n_splits. Обратите внимание, что порядок следования объектов в
# выборке может быть неслучайным, это может привести к смещенности кросс-валидационной оценки. Чтобы устранить
# такой эффект, объекты выборки случайно перемешивают перед разбиением на блоки. Для перемешивания достаточно передать
# генератору KFold параметр shuffle=True.
# Вычислить качество на всех разбиениях можно при помощи функции sklearn.model_selection.cross_val_score.
# В качестве параметра estimator передается классификатор, в качестве параметра cv — генератор разбиений
# с предыдущего шага. С помощью параметра scoring можно задавать меру качества, по умолчанию в задачах классификации
# используется доля верных ответов (accuracy). Результатом является массив, значения которого нужно усреднить.
#
# Приведение признаков к одному масштабу можно делать с помощью функции sklearn.preprocessing.scale,
# которой на вход необходимо подать матрицу признаков и получить масштабированную матрицу,
# в которой каждый столбец имеет нулевое среднее значение и единичное стандартное отклонение.


# Оценку качества необходимо провести методом кросс-валидации по 5 блокам (5-fold).
# Создайте генератор разбиений, который перемешивает выборку перед формированием блоков (shuffle=True).
# Для воспроизводимости результата, создавайте генератор KFold с фиксированным параметром random_state=42.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

y = data['type']
X = data[['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                                         'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                                         'Color intensity', 'Hue', 'diluted wines', 'Proline']]
# В качестве меры качества используйте долю верных ответов (accuracy).
# Найдите точность классификации на кросс-валидации для метода k ближайших соседей
# (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50.

scores1 = list()
for k in range(1, 51):
    model = KNeighborsClassifier(n_neighbors=k)
    scores1.append(cross_val_score(model,
                                  X,
                                  y,
                                  cv=kf,
                                  scoring='accuracy'))
res = pandas.DataFrame(scores1, range(1, 51)).mean(axis=1).sort_values(ascending=False)
accuracy = res.head(1)
print(accuracy)
# При каком k получилось оптимальное качество?
# 1
# Чему оно равно (число в интервале от 0 до 1)?
# 0.730476 = 0.73

# Произведите масштабирование признаков с помощью функции sklearn.preprocessing.scale.
# Снова найдите оптимальное k на кросс-валидации.
X = scale(X)
scores2 = list()
for k in range(1, 51):
    model = KNeighborsClassifier(n_neighbors=k)
    scores2.append(cross_val_score(model,
                                  X,
                                  y,
                                  cv=kf,
                                  scoring='accuracy'))
res2 = pandas.DataFrame(scores2, range(1, 51)).mean(axis=1).sort_values(ascending=False)
accuracy2 = res2.head(1)
print(accuracy2)
# Какое значение k получилось оптимальным после приведения признаков к одному масштабу?
# 29
# Чему оно равно (число в интервале от 0 до 1)?
# 0.977619 = 0.98
# Помогло ли масштабирование признаков?
# Да

