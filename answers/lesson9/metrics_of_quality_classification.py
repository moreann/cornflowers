import pandas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve


# Загрузите файл classification.csv. В нем записаны
# истинные классы объектов выборки (колонка true) и
# ответы некоторого классификатора (колонка pred).
cl = pandas.read_csv('classification.csv')

#      Заполните таблицу ошибок классификации:
#
# 	Actual Positive	Actual Negative
# Predicted Positive	TP	FP
# Predicted Negative	FN	TN
#
# Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям.
# Например, FP — это количество объектов, имеющих класс 0, но отнесенных алгоритмом к классу 1.
# Ответ в данном вопросе — четыре числа через пробел.

tp = 0  # истинно-положительные
fp = 0  # ложно-положительные
fn = 0  # ложно-отрицательные
tn = 0  # истинно-отрцитальные

for i in range(len(cl)):
    if cl['true'][i] == cl['pred'][i] and cl['true'][i] == 1:
        tp = tp + 1
    elif cl['true'][i] == 0 and cl['pred'][i] == 1:
        fp = fp + 1
    elif cl['true'][i] == 1 and cl['pred'][i] == 0:
        fn = fn + 1
    elif cl['true'][i] == cl['pred'][i] and cl['true'][i] == 0:
        tn = tn + 1
# 43 34 59 64

# Посчитайте основные метрики качества классификатора:
#     Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
#     Precision (точность) — sklearn.metrics.precision_score
#     Recall (полнота) — sklearn.metrics.recall_score
#     F-мера — sklearn.metrics.f1_score
# В качестве ответа укажите эти четыре числа через пробел.

accuracy = accuracy_score(cl['true'], cl['pred'])
precision = precision_score(cl['true'], cl['pred'])
recall = recall_score(cl['true'], cl['pred'])
f = f1_score(cl['true'], cl['pred'])
# print(round(accuracy, 3), ' ', round(precision, 3), ' ', round(recall, 3), ' ', round(f, 3))
# 0.535 0.558 0.422 0.48

# 4. Имеется четыре обученных классификатора. В файле scores.csv записаны истинные классы и
# значения степени принадлежности положительному классу для каждого классификатора на некоторой выборке:
#     для логистической регрессии — вероятность положительного класса (колонка score_logreg),
#     для SVM — отступ от разделяющей поверхности (колонка score_svm),
#     для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
#     для решающего дерева — доля положительных объектов в листе (колонка score_tree).
# Загрузите этот файл.

sc = pandas.read_csv('scores.csv')

# Посчитайте площадь под ROC-кривой для каждого классификатора.
# Какой классификатор имеет наибольшее значение метрики AUC-ROC (укажите название столбца)?
# Воспользуйтесь функцией sklearn.metrics.roc_auc_score.

log_roc = roc_auc_score(sc['true'], sc['score_logreg'])
svm_roc = roc_auc_score(sc['true'], sc['score_svm'])
knn_roc = roc_auc_score(sc['true'], sc['score_knn'])
tree_roc = roc_auc_score(sc['true'], sc['score_tree'])
# print(round(log_roc, 3), round(svm_roc, 3), round(knn_roc, 3), round(tree_roc, 3))
# 0.719 0.709 0.635 0.692
# max = 0.719 score_logreg

# Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?
# Чтобы получить ответ на этот вопрос, найдите все точки precision-recall-кривой с помощью
# функции sklearn.metrics.precision_recall_curve. Она возвращает три массива:
# precision, recall, thresholds. В них записаны точность и полнота при определенных порогах,
# указанных в массиве thresholds.
# Найдите максимальной значение точности среди тех записей, для которых полнота не меньше, чем 0.7.
pts_logreg = precision_recall_curve(sc['true'], sc['score_logreg'])
pts_svm = precision_recall_curve(sc['true'], sc['score_svm'])
pts_knn = precision_recall_curve(sc['true'], sc['score_knn'])
pts_tree = precision_recall_curve(sc['true'], sc['score_tree'])

logreg_max_prec = pts_logreg[0][(pts_logreg[1] >= 0.7)].max()
svm_max_prec = pts_svm[0][(pts_svm[1] >= 0.7)].max()
knn_max_prec = pts_knn[0][(pts_knn[1] >= 0.7)].max()
tree_max_prec = pts_tree[0][(pts_tree[1] >= 0.7)].max()
print(logreg_max_prec, svm_max_prec, knn_max_prec, tree_max_prec)
# score_tree
