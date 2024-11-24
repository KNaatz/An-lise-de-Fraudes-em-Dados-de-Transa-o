import zipfile
import os

caminho_zip = 'C:/Users/kauan/OneDrive/Área de Trabalho/Análise de Fraudes em Dados de Transação/archive.zip'
diretorio_destino = 'C:/Users/kauan/OneDrive/Área de Trabalho/Análise de Fraudes em Dados de Transação'

with zipfile.ZipFile (caminho_zip, 'r') as zip_ref:
    zip_ref.extractall(diretorio_destino)

print('Extração concluida')
