# -*- coding: utf-8 -*-

import numpy as np
import pandas as pnds
import re

from matplotlib import pyplot as pp
from scipy.stats import ttest_ind

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
pp.hist(age_stats, bins=20)
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
pp.hist(fare_stats, bins=20)

#wykorzystamy mode
nan_dict['Fare'] = raw_data_training['Fare'].mode()[0]

#Czas wyczyscic dane!
data_training = fill_nan(raw_data_training, nan_dict)
data_test = fill_nan(raw_data_test, nan_dict)

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

#a jak z wiekiem? Rozklady sa jakies inne?
print('Age')
age_survived = data_training['Age'].values[data_training['Survived'].values == 1]
age_dead = data_training['Age'].values[data_training['Survived'].values == 0]
pp.figure(2)
pp.hist([age_survived, age_dead], bins=10)
print(ttest_ind(age_survived, age_dead))
#11% szansa, ze nie ma roznicy w rozkladzie wieku obu grup

#niektore tytuly maja 100% lub 0% przezywalnosc. Sprawdzmy ile osob nosi dany
#tytul
print(data_training['Title'].value_counts())
#pominiemy tytuly z iloscia wystapien < 6. Pozostanie nam:
#Mr, Miss, Mrs, Master, Dr, Rev. 
data_training['Title'] = encode_title(data_training)
data_test['Title'] = encode_title(data_test)


X_train, y_train = x_y_split(data_training)
X_test, _ = x_y_split(data_test)