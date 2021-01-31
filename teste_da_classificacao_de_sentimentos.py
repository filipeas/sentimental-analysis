import pandas as pd
import string
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
import random
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

pln = spacy.load('pt')

pontuacoes = string.punctuation
stop_words = STOP_WORDS

def preprocessamento(texto):
    texto = texto.lower()
    documento = pln(texto)
    lista = []
    
    for token in documento:
        #lista.append(token.text)
        lista.append(token.lemma_)
    
    lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in pontuacoes]
    lista = ' ' . join([str(elemento) for elemento in lista if not elemento.isdigit()])
    
    return lista

modelo_carregado = spacy.load("modelo")

texto_positivo = "eu adoro a cor do seus olhos"
texto_positivo = preprocessamento(texto_positivo)
previsao = modelo_carregado(texto_positivo)
previsao.cats


texto_negativo = "estou com medo dele"
texto_negativo = preprocessamento(texto_negativo)
previsao = modelo_carregado(texto_negativo)
previsao.cats


base_dados = pd.read_csv('base_treinamento.txt', encoding='utf-8')
base_dados['texto'] = base_dados['texto'].apply(preprocessamento)
previsoes = []
for texto in base_dados['texto']:
    #print(texto)
    previsao = modelo_carregado(texto)
    previsoes.append(previsao.cats)

previsoes_final = []
for previsao in previsoes:
    if previsao['ALEGRIA'] > previsao['MEDO']:
        previsoes_final.append('alegria')
    else:
        previsoes_final.append('medo')
previsoes_final = np.array(previsoes_final)

respostas_reais = base_dados['emocao'].values

accuracy_score(respostas_reais, previsoes_final)
cm = confusion_matrix(respostas_reais, previsoes_final)

base_dados_teste = pd.read_csv('base_teste.txt', encoding='utf-8')
base_dados_teste['texto'] = base_dados_teste['texto'].apply(preprocessamento)
previsoes = []
for texto in base_dados_teste['texto']:
    #print(texto)
    previsao = modelo_carregado(texto)
    previsoes.append(previsao.cats)

previsoes_final = []
for previsao in previsoes:
    if previsao['ALEGRIA'] > previsao['MEDO']:
        previsoes_final.append('alegria')
    else:
        previsoes_final.append('medo')
previsoes_final = np.array(previsoes_final)
respostas_reais = base_dados_teste['emocao'].values

accuracy_score(respostas_reais, previsoes_final)
cm = confusion_matrix(respostas_reais, previsoes_final)
    
    
    
    
    
    
    
    
    
    