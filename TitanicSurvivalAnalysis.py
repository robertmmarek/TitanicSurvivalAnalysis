# -*- coding: utf-8 -*-

import numpy as np
import pandas as pnds
import re
import os

from matplotlib import pyplot as pp
from scipy.stats import ttest_ind
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC


def encode_column(data_frame, 
                  column_name, 
                  label_encoder=None, 
                  one_hot_encoder=None):
    
    data = data_frame.copy()
    if label_encoder == None:
        label_encoder = LabelEncoder()
        label_encoder.fit(data.loc[:, column_name])
        
    data.loc[:, column_name] = label_encoder.transform(data.loc[:, column_name])
    
    if one_hot_encoder == None:
        one_hot_encoder = OneHotEncoder(sparse=False)
        one_hot_encoder.fit(data.loc[:, column_name].values.reshape(-1, 1))
        
    encoded = one_hot_encoder.transform(data.loc[:, column_name].values.reshape(-1,1))
    
    for i in range(0, encoded.shape[1]-1):
        data[column_name+str(i)] = encoded[:, i]
        
    data = data.drop(column_name, axis=1)
    return data, label_encoder, one_hot_encoder
    

def extract_title(string):
    search_res = re.search(r'[\w]+,\s+([\w]+)\.', string)
    if search_res != None:
        return search_res.group(1)
    else:
        return ''

def is_married_woman(data_frame):
    data = data_frame.copy()
    is_female = lambda x: x == 'female' 
    is_married = lambda x: x == 'Mrs'
    binarize_dict = {True: 1, False: 0}
    binarize = lambda x: binarize_dict[x]
    
    data.loc[:, 'MarriedWoman'] = data['Title'].apply(is_married) & data['Sex'].apply(is_female)
    data['MarriedWoman'] = data['MarriedWoman'].apply(binarize)
    return data['MarriedWoman']

def is_alone(data_frame):
    data = data_frame.copy()
    binarize_dict = {True: 1, False: 0}
    binarize = lambda x: binarize_dict[x]
    data.loc[:, 'IsAlone'] = (data['SibSp'] == 0) & (data['Parch'] == 0)
    data['IsAlone'] = data['IsAlone'].apply(binarize)
    return data['IsAlone']

def has_children(data_frame):
    data = data_frame.copy()
    has_child_array = (data['Parch'] > 0.) & (data['Age'] >= 18.)
    h_children = lambda x: 1.0 if x else 0.0
    data.loc[:, 'HasChildren'] = has_child_array
    data['HasChildren'] = data['HasChildren'].apply(h_children)  
    temp_array = data.loc[:, 'HasChildren']
    temp_array = np.array([2. if age < 18. else x for x, age in zip(temp_array,
                                                                    data.loc[:, 'Age'])])
    data.loc[:, 'HasChildren'] = temp_array
    return data['HasChildren']


def fill_nan(data_frame, nan_replacement_dict=None): 
    data = data_frame.copy()
    for key, value in nan_replacement_dict.items():
        data[key] = data[key].fillna(value=value)       
    return data

def x_y_split(data_frame):
    data = data_frame.copy()
    columns = list(data.columns.values)
    if 'Survived' in columns:
        columns.remove('Survived')
        return data[columns], data['Survived']
    else:
        return data[columns], None
    
def encode_title(data_frame):
    data = data_frame.copy()
#    Mr, Miss, Mrs, Master, Dr, Rev. 
    encode_dict = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6}
    encode = lambda x: encode_dict[x] if x in encode_dict.keys() else 0
    data['Title'] = data['Title'].apply(encode)
    
    return data['Title']

raw_data_training = pnds.read_csv(r'train.csv')
raw_data_test = pnds.read_csv(r'test.csv')

"""Przygotowanie danych"""
#PassengerId jest zbednym parametrem
raw_data_training = raw_data_training.drop(['PassengerId'], axis=1)
test_passenger_id = raw_data_test.loc[:, 'PassengerId'].values.astype('int')
raw_data_test = raw_data_test.drop(['PassengerId'], axis=1)

#Z jednej strony parametr Cabin zawiera wiele pustych pozycji, z drugiej strony
#bezposrednio jest zwiazany z pokladem na ktorym znajdowala sie osoba
#https://www.encyclopedia-titanica.org/titanic-deckplans/
#pominiemy ten parametr w tej analizie

raw_data_training = raw_data_training.drop(['Cabin'], axis=1)
raw_data_test = raw_data_test.drop(['Cabin'], axis=1)

#Niespecjalnie wiadomo rowniez co zrobic z numerem biletu.

raw_data_training = raw_data_training.drop(['Ticket'], axis=1)
raw_data_test = raw_data_test.drop(['Ticket'], axis=1)

#Wyodrebnimy takze tytuly kazdego z pasazerow. Byc moze bedziemy w stanie
#zauwazyc jakas ciekawa zaleznosc zwiazana z tytulem. Z pewnoscia bedziemy
#mogli stwierdzic ktora kobieta byla mezatka

raw_data_training['Title'] = raw_data_training['Name'].apply(extract_title)
raw_data_test['Title'] = raw_data_test['Name'].apply(extract_title)

#Usuniemy niepotrzebna juz kolumne z imieniem i nazwiskiem

raw_data_training = raw_data_training.drop(['Name'], axis=1)
raw_data_test = raw_data_test.drop(['Name'], axis=1)

#Stworzymy dodatkowa ceche - bycie mezatka
raw_data_training['MarriedWoman'] = is_married_woman(raw_data_training)
raw_data_test['MarriedWoman'] = is_married_woman(raw_data_test)

#W pozostalych danych sa wieksze braki. Czas je uzupelnic
nan_dict = {}

#dla klasy - moda
nan_dict['Pclass'] = raw_data_training['Pclass'].mode()[0]

#dla miejsca wejscia na poklad - rowniez moda
nan_dict['Embarked'] = raw_data_training['Embarked'].mode()[0]

#Jaka jest struktura wieku na titanicu?
pp.figure(0)
age_stats = np.array(raw_data_training['Age'].dropna().loc[:])
pp.hist(age_stats, bins=20, normed=True)
pp.xlabel('Age')
pp.ylabel('% of specific age')
pp.title('Age distribution')
pp.show()

age_mode = raw_data_training['Age'].mode()[0]
age_median = np.median(raw_data_training['Age'].dropna().loc[:])
age_mean = np.mean(raw_data_training['Age'].dropna().loc[:])

#sprobujemy wykorzystac mode
nan_dict['Age'] = age_mode

#ilosc rodzenstwa oraz ilosc dzieci - moda
nan_dict['SibSp'] = raw_data_training['SibSp'].mode()[0]
nan_dict['Parch'] = raw_data_training['Parch'].mode()[0]

#Jaka jest struktura ceny biletu?
pp.figure(1)
fare_stats = np.array(raw_data_training['Fare'].dropna().loc[:])
pp.hist(fare_stats, bins=20, normed=True)
pp.xlabel('Fare')
pp.ylabel('% of specific fare')
pp.title('Fare distribution')
pp.show()

#wykorzystamy mode
nan_dict['Fare'] = raw_data_training['Fare'].mode()[0]

#Czas wyczyscic dane!
data_training = fill_nan(raw_data_training, nan_dict)
data_test = fill_nan(raw_data_test, nan_dict)

#dodajmy nastepujaca ceche - jezeli ktos ma wiecej niz 18 lat 
#uznajemy parch za ilosc dzieci. 
#dodatkowo, kodujemy dzieci jako '2'

data_training['HasChildren'] = has_children(data_training)
data_test['HasChildren'] = has_children(data_test)

#dodamy jeszcze jedna ceche - czy jest samotny na statku?
data_training['IsAlone'] = is_alone(data_training)
data_test['IsAlone'] = is_alone(data_test)

#plec trzeba zbinaryzowac
sex_dict = {'female': 0, 'male': 1}
binarize_sex = lambda x: sex_dict[x]
data_training['Sex'] = data_training['Sex'].apply(binarize_sex)
data_test['Sex'] = data_test['Sex'].apply(binarize_sex)

"""Przeanalizujmy to, co mamy"""
#jak wyglada przezywalnosc wedlug plci?
print('Przezywalnosc wzgledem plci')
print(data_training[['Survived', 'Sex']].groupby(['Sex']).mean())

#jak wyglada przezywalnosc wedlug klasy?
print('Pclass')
print(data_training[['Survived', 'Pclass']].groupby(['Pclass']).mean())

#jak wyglada przezywalnosc wedlug ilosci rodzenstwa/małżonka?
print('SibSp')
print(data_training[['Survived', 'SibSp']].groupby(['SibSp']).mean())

#a jak w przypadku rodzicow/dzieci?
print('Parch')
print(data_training[['Survived', 'Parch']].groupby(['Parch']).mean())

#czy mezatki radza sobie lepiej?
print('MarriedWoman')
print(data_training[['Survived', 'MarriedWoman']].groupby(['MarriedWoman']).mean())

#co z osobami samotnymi?
print('isAlone')
print(data_training[['Survived', 'IsAlone']].groupby(['IsAlone']).mean())

#co z tytulami?
print('Title')
print(data_training[['Survived', 'Title']].groupby(['Title']).mean())

#czy posiadanie dzieci sprzyja przezyciu?
print('HasChildren')
print(data_training[['Survived', 'HasChildren']].groupby(['HasChildren']).mean())

#a jak z wiekiem? Rozklady sa jakies inne?
print('Age')
age_survived = data_training['Age'].values[data_training['Survived'].values == 1]
age_dead = data_training['Age'].values[data_training['Survived'].values == 0]

pp.figure(2)
pp.hist([age_survived, age_dead], 
        bins=10, 
        normed=True, 
        color='br', 
        label=['Survived', 'Dead'])
pp.xlabel('Age')
pp.ylabel('% of specific Age')
pp.title('Survived vs dead age distribution')
pp.legend()
print(ttest_ind(age_survived, age_dead))

#11% szansa, ze nie ma roznicy w rozkladzie wieku obu grup

#niektore tytuly maja 100% lub 0% przezywalnosc. Sprawdzmy ile osob nosi dany
#tytul
print(data_training['Title'].value_counts())
#pominiemy tytuly z iloscia wystapien < 6. Pozostanie nam:
#Mr, Miss, Mrs, Master, Dr, Rev. 
data_training['Title'] = encode_title(data_training)
data_test['Title'] = encode_title(data_test)

#zakodujmy jeszcze odpowiednio niektore kolumny
data_training, label_enc, oh_enc = encode_column(data_training, 'Pclass')
data_test, _, _ = encode_column(data_test, 'Pclass')

data_training, label_enc, oh_enc = encode_column(data_training, 'Embarked')
data_test, _, _ = encode_column(data_test, 'Embarked')

data_training, label_enc, oh_enc = encode_column(data_training, 'Title')
data_test, _, _ = encode_column(data_test, 'Title')

data_training, label_enc, oh_enc = encode_column(data_training, 'HasChildren')
data_test, _, _ = encode_column(data_test, 'HasChildren')

X_train, y_train = x_y_split(data_training)
X_test, _ = x_y_split(data_test)

"""Budujemy klasyfikator!"""
def forest_backward_selection(forest, X, y, threshold):
    while True:
        columns = list(X.columns.values)
        forest.fit(X, y)
        importances = list(forest.feature_importances_)
        
        if min(importances) >= threshold:
            break
        else:
            index_to_remove = importances.index(min(importances))
            columns.pop(index_to_remove)
            X = X[columns]
        
    return X, columns

#mean_scores = []
#train_scores = []
#index_train = []
#for i in np.linspace(2., 90, num=50):
#    forest = RandomForestClassifier(n_estimators=int(i),
#                                    min_samples_split=13,
#                                    min_samples_leaf=5,
#                                    random_state=0)
#    
#    #svc = SVC(probability=False, random_state=0)
#    
#    _, selected_columns = forest_backward_selection(forest, X_train, y_train, 0.)
#    
#    X_train = X_train[selected_columns]
#    X_test = X_test[selected_columns]
#    
#    kfold = StratifiedKFold(n_splits=3, random_state=0)
#    X = X_train.values.astype('float')
#    y = y_train.values.astype('float')
#    scores = []
##    for train, test in kfold.split(X, y):
##        forest.fit(X[train], y[train])
##        scores += [forest.score(X[test], y[test])]
#    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.3, random_state=0)
#    forest.fit(X_tr, y_tr)
#    #svc.fit(X_tr, y_tr)
#    scores += [forest.score(X_tst, y_tst)]
#    mean_scores.append(np.mean(scores))
#    train_scores.append(forest.score(X_tr, y_tr))
#    print(mean_scores[-1], train_scores[-1])
#    index_train.append(i)
#pp.figure(3)
#pp.plot(index_train, mean_scores, color='r')
#pp.plot(index_train, train_scores, color='b')

forest = RandomForestClassifier(n_estimators=20,
                                min_samples_split=13,
                                min_samples_leaf=5,
                                random_state=0)

svc = SVC(probability=True)

X = X_train.values.astype('float')
y = y_train.values.astype('float')
forest.fit(X, y)
svc.fit(X, y)
X = X_test.values.astype('float')
result = forest.predict(X)

result_dataframe = pnds.DataFrame()
result_dataframe['PassengerId'] = test_passenger_id
result_dataframe['Survived'] = result.astype('int')


out_path = r'out.csv'
previous_dataframe = pnds.DataFrame()

try:
    previous_dataframe = pnds.read_csv(out_path)
    if all(previous_dataframe['Survived'].values == result_dataframe['Survived']):
        print('No change')
    else:
        print('Change!')
except Exception as error:
    print(error)

result_dataframe.to_csv(r'out.csv', index=False)
