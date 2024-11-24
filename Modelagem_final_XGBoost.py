# Modelagem testando XGBoost

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


# Carregando o dataset pré-processado anteriormente
df = pd.read_csv('análise_inicial.csv')

# Removendo colunas que não serão utilizadas na modelagem
df = df.drop(columns=['nameOrig', 'nameDest'])

# Separando as Features (X) da variável alvo (Y)
X = df.drop(columns=['isFraud'])
y = df['isFraud']

# Dividindo treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicando SMOTE para balancear os dados de treino
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Otimização
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],  # Corrigido aqui
    'scale_pos_weight': [1, 10, 50]
}

# Configurando o RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric='logloss'),
    param_distributions=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=2,
    n_iter=50  # Limite o número de combinações testadas
)

# Executando a busca
random_search.fit(X_train_res, y_train_res)

print("Melhores parâmetros encontrados:", random_search.best_params_)

# Melhores parâmetros encontrados pelo RandomizedSearchCV
best_params = random_search.best_params_

# Treinando o modelo final com os melhores parâmetros encontrados
best_params = {
    'subsample': 1.0,
    'scale_pos_weight': 10,
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.3,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss'  # Adicionando aqui para evitar warnings
}

# Criando o modelo otimizado
xgb_final = XGBClassifier(**best_params, random_state=42)

# Treinando o modelo otimizado
xgb_final.fit(X_train_res, y_train_res)

# Fazendo previsões no conjunto de teste
y_pred = xgb_final.predict(X_test)

# Avaliando a performance do modelo final
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

#configurações gráficas.

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='isFraud')
plt.title('Distribuição de Transações Fraudulentas')
plt.xlabel('Fraude (0 = Não, 1 = Sim)')
plt.ylabel('Contagem')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['amount'], bins=50, kde=True)
plt.title('Distribuição dos Valores das Transações')
plt.xlabel('Valor da Transação')
plt.ylabel('Frequência')
plt.show()

#Salvando os dados pré-processados
df.to_csv('Modelagem_XGBoost.csv', index=False)

    



