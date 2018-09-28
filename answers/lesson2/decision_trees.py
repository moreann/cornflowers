import pandas
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


data = pandas.read_csv('titanic.csv', index_col='PassengerId', )
data1 = pd.DataFrame(data=data, columns=['Pclass', 'Fare', 'Age', 'Sex', 'Survived'])
data1['Sex'] = data1['Sex'].map({'female': 1, 'male': 0})

data1.dropna(inplace=True)
y = data1[['Survived']]
x = data1[['Pclass', 'Fare', 'Age', 'Sex']]

clf = DecisionTreeClassifier(random_state=241)
clf.fit(x, y)
importances = clf.feature_importances_

print(importances)
