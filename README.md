#Análise de Fraudes em Dados de Transação

-Este projeto tem como objetivo analisar dados de transações financeiras para identificar possíveis fraudes usando técnicas de aprendizado de máquina. Utilizamos o dataset 'Synthetic Financial Datasets For Fraud Detection' disponível no Kaggle.
Nesta análise, buscamos identificar transações fraudulentas com alta precisão, minimizando falsos positivos. Para isso, foi utilizado uma técnica de análise inicial, em seguida, testes com a machine learning LogisticRegression. Devido a assertividade menos precisa, mudei a abordagem e a técnica final utilizada é a de machine learning XGBoost, qual obteve excelentes resultados.

O dataset utilizado foi 'Synthetic Financial Datasets For Fraud Detection' disponível no Kaggle. Inclui milhões de transações, das quais uma pequena porcentagem foi marcada como fraude.

As etapas dos processos são:

1. **Carregamento de Dados:** Carregamento e pré-processamento dos dados, divididos nos blocos de códigos denominados "extrair_dados.py" e "Verificação_arquivo.py"
2. **Análise Exploratória:** Exploração visual dos dados usando gráficos de distribuição de valores e classes, no bloco "análise_inicial.py"
3. **Primeira Modelagem:** Uso da técnica SMOTE para balanceamento dos dados e primeira modelagem com LogisticRegression.
4. **Modelagem Final:** Aplicação do modelo XGBoost com otimização de hiperparâmetros.
5. **Avaliação:** Avaliação da performance com imagens simplificadas de gráficos e curva ROC.

Apresentação dos principais resultados do projeto, incluindo gráficos gerados:
- Distribuição de Transações Fraudulentas.
- Distribuição de Valores das Transações.
- Curva ROC com AUC = 0.99.

Conclusão:

O modelo XGBoost se mostrou altamente eficaz na detecção de fraudes, atingindo um AUC de 0.99. O uso de técnicas como SMOTE foi fundamental para lidar com a desproporção de classes.

Bibliotecas utilizadas:

- Python 3.8+
- numpy
- pandas
- scikit-learn
- xgboost
-LogisticRegression
- matplotlib
- seaborn
