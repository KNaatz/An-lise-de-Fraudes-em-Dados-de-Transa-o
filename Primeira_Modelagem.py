#Modelagem

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#Carregando o dataset pré-processado anteriormente

df = pd.read_csv('análise_inicial.csv')

#Removendo colunas que não serão utilizadas na modelagem

df = df.drop(columns=['nameOrig', 'nameDest'])

#Separando as Features (X) da variável alvo (Y)
X = df.drop(columns=['isFraud'])
y = df['isFraud']

#Dividindo treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

#Modelo de regressão de Logística
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train_res, y_train_res)

#Previsão no conjunto de testes
y_pred = modelo.predict(X_test)

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))