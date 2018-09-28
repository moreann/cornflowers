import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

# Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах из файла
# salary-train.csv (либо его заархивированную версию salary-train.zip).
train = pd.read_csv('salary-train.csv')

# Проведите предобработку:
# Приведите тексты к нижнему регистру (text.lower()).
train['FullDescription'] = train.FullDescription.str.lower()

# Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова.
# Для такой замены в строке text подходит следующий вызов: re.sub('[^a-zA-Z0-9]', ' ', text).
# Также можно воспользоваться методом replace у DataFrame, чтобы сразу преобразовать все тексты:
train.replace({'FullDescription': r'[^a-zA-Z0-9]'}, {'FullDescription': ' '}, regex=True)

# Примените from sklearn.feature_extraction import DictVectorizer для преобразования текстов в векторы признаков.
# Оставьте только те слова, которые встречаются хотя бы в 5 объектах (параметр min_df у TfidfVectorizer).
td = TfidfVectorizer(min_df=5)
X_train_text = td.fit_transform(train['FullDescription'])

# Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'.
train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

# Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.
enc = DictVectorizer()
X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))

# Объедините все полученные признаки в одну матрицу "объекты-признаки".
# Обратите внимание, что матрицы для текстов и категориальных признаков являются разреженными.
# Для объединения их столбцов нужно воспользоваться функцией scipy.sparse.hstack.
X_train = hstack([X_train_text, X_train_categ])

# Обучите гребневую регрессию с параметрами alpha=1 и random_state=241.
# Целевая переменная записана в столбце SalaryNormalized.
y_train = train['SalaryNormalized']
gr = Ridge(alpha=1, random_state=241)
gr.fit(X_train, y_train)

# Постройте прогнозы для двух примеров из файла salary-test-mini.csv.
# Значения полученных прогнозов являются ответом на задание. Укажите их через пробел
test = pd.read_csv('salary-test-mini.csv')
test['FullDescription'] = test.FullDescription.str.lower()
test.replace({'FullDescription': r'[^a-zA-Z0-9]'}, {'FullDescription': ' '}, regex=True)

X_test_text = td.transform(test['FullDescription'])
test['LocationNormalized'].fillna('nan', inplace=True)
test['ContractTime'].fillna('nan', inplace=True)
X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test_text, X_test_categ])

y_test = gr.predict(X_test)
print(round(y_test[0], 2), round(y_test[1], 2))
