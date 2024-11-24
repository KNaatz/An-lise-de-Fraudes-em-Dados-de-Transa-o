import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

caminho_csv = 'C:/Users/kauan/OneDrive/Área de Trabalho/Análise de Fraudes em Dados de Transação/Factures.csv'
df = pd.read_csv(caminho_csv)

normalizar_colunas = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

scaler = MinMaxScaler()
df[normalizar_colunas] = scaler.fit_transform(df[normalizar_colunas])

#One-Hot Ecoding
df = pd.get_dummies(df, columns= ['type'])

print("visualização inicial dos dados")
print(df.head())

print("\nResumo Estatístico:")
print(df.describe())

print("\nDataframe:")
print(df.info())

#Verificação valores ausentes
print("Valores ausentes por coluna")
print(df.isnull().sum()) 

print("Normalização concluída. Visualização dos dados normalizados:")
print(df.head())

print("Codificação concluída. Visualização dos primeiros dados:")
print(df.head())

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
df.to_csv('análise_inicial.csv', index=False)