from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import numpy as np
import pandas as pd

# Для начала вам потребуется загрузить данные. В этом задании мы воспользуемся одним из датасетов,
# доступных в scikit-learn'е — 20 newsgroups. Для этого нужно воспользоваться модулем datasets:
# После выполнения этого кода массив с текстами будет находиться в поле newsgroups.data,
# номер класса — в поле newsgroups.target.
newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )
X = newsgroups.data
y = newsgroups.target

# Преобразование обучающей выборки нужно делать с помощью функции fit_transform, тестовой — с помощью transform.
# Вычислите TF-IDF-признаки для всех текстов. Обратите внимание, что в этом задании мы предлагаем вам
# вычислить TF-IDF по всем данным. При таком подходе получается, что
# признаки на обучающем множестве используют информацию из тестовой выборки — но такая ситуация вполне законна,
# поскольку мы не используем значения целевой переменной из теста.
# На практике нередко встречаются ситуации, когда признаки объектов тестовой выборки известны на момент обучения,
# и поэтому можно ими пользоваться при обучении алгоритма.

vectorizer = TfidfVectorizer()
X_fit = vectorizer.fit_transform(X)
X_test = vectorizer.transform(X)
# feature_mapping = vectorizer.get_feature_names()
# print(feature_mapping)

# Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5]
# для SVM с линейным ядром (kernel='linear') при помощи кросс-валидации по 5 блокам.
# Укажите параметр random_state=241 и для SVM, и для KFold.
# В качестве меры качества используйте долю верных ответов (accuracy).
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)

clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X_fit, y)

res2 = pd.DataFrame(data=gs.cv_results_)
res = res2.loc[res2['mean_test_score'].idxmax()]
C_min = res['param_C']
# mean_validation_score — оценка качества по кросс-валидации
# param_C — значения параметра С

# Обучите SVM по всей выборке с оптимальным параметром C, найденным на предыдущем шаге.
clf1 = SVC(C=C_min, kernel='linear', random_state=241)
clf1.fit(X_fit, y)

# Найдите 10 слов с наибольшим абсолютным значением веса (веса хранятся в поле coef_ у svm.SVC).
# Они являются ответом на это задание.
# Укажите эти слова через запятую или пробел, в нижнем регистре, в лексикографическом порядке.
word_indexes = np.argsort(np.abs(clf1.coef_.toarray()[0]))[-10:]
words = [vectorizer.get_feature_names()[i] for i in word_indexes]
words.sort()
print(words)

