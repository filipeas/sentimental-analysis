import pandas as pd
import string
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
import random
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

base_dados = pd.read_csv('base_treinamento.txt', encoding='utf-8')

sns.countplot(base_dados['emocao'], label='contagem')

pontuacoes = string.punctuation
stop_words = STOP_WORDS

pln = spacy.load('pt')

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

    
#teste = preprocessamento('Estou 1 10 aPrendendo processamento de linguagem natural, Curso em Curitiba')

base_dados['texto'] = base_dados['texto'].apply(preprocessamento)

base_dados_final = []
for texto, emocao in zip(base_dados['texto'], base_dados['emocao']):
    #print(texto, emocao)
    if emocao == 'alegria':
        dic = ({'ALEGRIA': True, 'MEDO': False})
    elif emocao == 'medo':
        dic = ({'ALEGRIA': False, 'MEDO': True})
    
    base_dados_final.append([texto, dic.copy()])


modelo = spacy.blank('pt')
categorias = modelo.create_pipe('textcat')
categorias.add_label("ALEGRIA")
categorias.add_label("MEDO")
modelo.add_pipe(categorias)
historico = []

modelo.begin_training()
for epoca in range(1000):
    random.shuffle(base_dados_final)
    losses = {}
    for batch in spacy.util.minibatch(base_dados_final, 30):
        textos = [modelo(texto) for texto, entities in batch]
        annotations = [{'cats': entities} for texto, entities in batch]
        modelo.update(textos, annotations, losses = losses)
    if epoca % 100 == 0:
        print(losses)
        historico.append(losses)


historico_loss = []
for i in historico:
    historico_loss.append(i.get('textcat'))
historico_loss = np.array(historico_loss)

plt.plot(historico_loss)
plt.title('Progressão do erro')
plt.xlabel('Épocas')
plt.ylabel('Erro')

modelo.to_disk("modelo")





