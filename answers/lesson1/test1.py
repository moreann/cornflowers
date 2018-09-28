import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

all_people = len(data)

female = 0
male = 0
x = 0

for i in data['Sex']:
    if i == 'female':
        female = female + 1
    elif i == 'male':
        male = male + 1
    else:
        x = x + 1

# print('female = ' + str(female), 'male = ' + str(male), 'x=' + str(x))

live = data['Survived'].value_counts()[1]


# print(all_people)

p = round(100*(live/all_people), 2)
# print(p)

luxury = data['Pclass'].value_counts()[1]
a = round(100*luxury/all_people, 2)
# print(a)

# print(round(data['Age'].describe(), 2))
# print(round(data['Age'].median(), 2))

correlation = data['SibSp'].corr(data['Parch'], method='pearson')
correlation = round(correlation, 2)
# print(correlation)

# female_name = data['Name'][data['Sex'] == 'female']
#print(female_name)

# a = list()
# a.append(re.search('((Mrs\. |Miss\. |Lady\. |Mme\. )(?:.*\()(?P<maiden_name>\w+))'
#               '|((Mrs\. |Miss\. |Lady\.  |Mme\. )(?P<name>\w+))', female_name))
full_names = pandas.Series(data['Name'][data['Sex'] == 'female'])
reg_str = '((Mrs\. |Miss\. )(?:.*\()(?P<maiden_name>\w+))|((Mrs\. |Miss\. )(?P<name>\w+))'
matches = full_names.str.extract(reg_str, expand=False)
names = matches['name'].combine_first(matches['maiden_name'])
print(names)
print(names.mode())