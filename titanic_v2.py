# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 23:17:51 2025

@author: victor
"""
import pandas as pd
import seaborn as sns
import sklearn as sklearn
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##############################

df_train = pd.read_csv('C:/Users/victo/Downloads/titanic/train.csv')
df_teste = pd.read_csv('C:/Users/victo/Downloads/titanic/test.csv')
train = df_train
teste = df_teste

############################

train['Sex'] = train['Sex'].map(lambda x: 1 if x == 'female' else 0)
mean = train['Age'].mean()
train['Age'] = train['Age'].fillna(mean)
train_clean = train.drop(columns = ['Name', 'Ticket', 'Cabin','Embarked'])
teste['Sex'] = teste['Sex'].map(lambda x: 1 if x == 'female' else 0)
teste['Age'] = teste['Age'].fillna(mean)
mean2 = teste['Fare'].mean()
teste['Fare'] = teste['Fare'].fillna(mean2)
teste_clean = teste.drop(columns = ['Name', 'Ticket', 'Cabin','Embarked'])

###########################

x = train_clean[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = train_clean['Survived']
x_teste = teste_clean[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
model = LogisticRegression()
model.fit(x,y)
model.score(x,y)
y_pred = model.predict(x_teste)
df_y_pred = pd.DataFrame(y_pred, columns=['Prevision_Survived'])
df_y_pred.to_csv('previsoes_teste.csv', index=False)