import pandas as pd
import string
import spacy
import random
import seaborn as sns
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

base_treinamento = pd.read_csv('Train50.csv', delimiter = ';')
sns.countplot(base_treinamento['sentiment'], label = 'Contagem')
base_treinamento.drop(['id', 'tweet_date', 'query_used'], axis = 1, inplace=True)
base_teste = pd.read_csv('Test.csv', delimiter = ';')
sns.countplot(base_teste['sentiment'], label = 'Contagem')
base_teste.drop(['id', 'tweet_date', 'query_used'], axis = 1, inplace=True)

sns.heatmap(pd.isnull(base_treinamento))
sns.heatmap(pd.isnull(base_teste))

pln = spacy.load('pt')
stop_words = spacy.lang.pt.stop_words.STOP_WORDS

def preprocessamento(texto):
    # Letras minusculas
    texto = texto.lower()
    texto = re.sub(r"@[A-Za-z0-9$-_@.&+]+", ' ', texto)
    
    # URLs
    texto = re.sub(r"https?://[A-Za-z0-9./]+", ' ', texto)
    
    # Espaços em branco
    texto = re.sub(r" +", ' ', texto)
    
    # Emoticons
    lista_emocoes = {':)': 'emocaopositiva',
                     ':d': 'emocaopositiva',
                     ':(': 'emocaonegativa'}
    for emocao in lista_emocoes:
        texto = texto.replace(emocao, lista_emocoes[emocao])
    
    # Lematização
    documento = pln(texto)
    lista = []
    for token in documento:
        lista.append(token.lemma_)
    
    # Stop Words e pontuações
    lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in string.punctuation]
    lista = ' ' . join([str(elemento) for elemento in lista if not elemento.isdigit()])
    
    return lista

# texto_teste = '@behin_d_curtain Para mim, :D ou :d para :( http://www.google.com é precisamente o contrário :) Vem a chuva e vem a boa disposição :)'
# resultado = preprocessamento(texto_teste)

base_treinamento['tweet_text'] = base_treinamento['tweet_text'].apply(preprocessamento)
base_teste['tweet_text'] = base_teste['tweet_text'].apply(preprocessamento)

base_dados_treinamento_final = []
for texto, emocao in zip(base_treinamento['tweet_text'], base_treinamento['sentiment']):
    if emocao == 1:
        dic = ({'POSITIVO': True, 'NEGATIVO': False})
    elif emocao == 0:
        dic = ({'POSITIVO': False, 'NEGATIVO': True})
    base_dados_treinamento_final.append([texto, dic.copy()])

modelo = spacy.blank('pt')
categorias = modelo.create_pipe('textcat')
categorias.add_label("POSITIVO")
categorias.add_label("NEGATIVO")
modelo.add_pipe(categorias)
historico = []

modelo.begin_training()
for epoca in range(20):
    random.shuffle(base_dados_treinamento_final)
    losses = {}
    for batch in spacy.util.minibatch(base_dados_treinamento_final, 512):
        textos = [modelo(texto) for texto, entities in batch]
        annotations = [{'cats': entities} for texto, entities in batch]
        modelo.update(textos, annotations, losses = losses)
    if epoca % 5 == 0:
        print(losses)
        historico.append(losses)

historico_loss = []
for i in historico:
    historico_loss.append(i.get('textcat'))
historico_loss = np.array(historico_loss)

plt.plot(historico_loss)
plt.title('Progressão do erro')
plt.xlabel('Batches')
plt.ylabel('Erro')

modelo.to_disk("modelo_twitter")

texto_positivo = base_teste['tweet_text'][21]
previsao = modelo(texto_positivo)
texto_positivo = 'eu gosto muito de você'
texto_positivo = preprocessamento(texto_positivo)
previsao = modelo(texto_positivo)
texto_negativo = base_teste['tweet_text'][4000]
previsao = modelo(texto_negativo)

previsoes = []
for texto in base_treinamento['tweet_text']:
    previsao = modelo(texto)
    previsoes.append(previsao.cats)

previsoes_final = []
for previsao in previsoes:
    if previsao['POSITIVO'] > previsao['NEGATIVO']:
        previsoes_final.append(1)
    else:
        previsoes_final.append(0)
previsoes_final = np.array(previsoes_final)

respostas_reais = base_treinamento['sentiment'].values

accuracy_score(respostas_reais, previsoes_final)
cm = confusion_matrix(respostas_reais, previsoes_final)
sns.heatmap(cm, annot = True)

previsoes = []
for texto in base_teste['tweet_text']:
    previsao = modelo(texto)
    previsoes.append(previsao.cats)

previsoes_final = []
for previsao in previsoes:
    if previsao['POSITIVO'] > previsao['NEGATIVO']:
        previsoes_final.append(1)
    else:
        previsoes_final.append(0)
previsoes_final = np.array(previsoes_final)
respostas_reais = base_teste['sentiment'].values

accuracy_score(respostas_reais, previsoes_final)
cm = confusion_matrix(respostas_reais, previsoes_final)
sns.heatmap(cm, annot = True)
