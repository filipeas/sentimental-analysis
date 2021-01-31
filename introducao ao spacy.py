import spacy
import nltk
from spacy import displacy
from spacy.lang.pt.stop_words import STOP_WORDS

pln = spacy.load('pt')

documento = pln('Estou aprendendo processamento de linguagem natural, curso em Curitiba')
for token in documento:
    print(token.text, token.pos_)

for token in documento:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)

for token in documento:
    if token.pos_ == 'PROPN':
        print(token.text)



for token in documento:
    print(token.text, token.lemma_)

doc = pln('encontrei encontraram encontrarão encontrariam')
[token.lemma_ for token in doc]

nltk.download('rslp')

stemmer = nltk.stem.RSLPStemmer()
stemmer.stem('aprendendo')

for token in documento:
    print(token.text, token.lemma_, stemmer.stem(token.text))


texto = 'A IBM é uma empresa dos Estados Unidos voltada para a área de informática. Sua sede no Brasil fica em São Paulo e a receita em 2018 foi de aproximadamente 320 bilhões de reais'
documento = pln(texto)

for entidade in documento.ents:
    print(entidade.text, entidade.label_)

displacy.render(documento, style = 'ent', jupyter= True)

texto = 'Bill Gates nasceu em Seattle em 28/10/1955 e foi o criador da Microsoft'
documento = pln(texto)
for entidade in documento.ents:
    print(entidade.text, entidade.label_)

for entidade in documento.ents:
    if entidade.label_ == 'PER':
        print(entidade.text)

print(STOP_WORDS)
len(STOP_WORDS)
pln.vocab['ir'].is_stop
pln.vocab['caminhar'].is_stop

documento = pln('Estou aprendendo processamento de linguagem natural, curso em Curitiba')
for token in documento:
    if not pln.vocab[token.text].is_stop:
        print(token.text)



documento = pln('reserve uma passagem saindo de Guarulhos e chegando em Curitiba')
origem = documento[5]
destino = documento[9]
origem, destino

list(origem.ancestors)
list(destino.ancestors)

documento[0].is_ancestor(documento[2])


documento = pln('Reserva de uma mesa para o restaurante e de um táxi para o hotel')
tarefas = documento[3], documento[10]
locais = documento[6], documento[13]
for local in locais:
    for objeto in local.ancestors:
        if objeto in tarefas:
            print("reserva de {} é para o {}". format(objeto, local))
            break

list(documento[6].children)

displacy.serve(documento, style='dep')
displacy.render(documento, style='dep', jupyter=False, options={'distance':90})

list(documento[3].ancestors)
list(documento[3].children)

documento = pln('Que locais podemos visitar em Curitiba e para ficar em Guarulhos?')
lugares = [token for token in documento if token.pos_ == 'PROPN']
acoes = [token for token in documento if token.pos_ == 'VERB']

for local in lugares:
    for acao in local.ancestors:
        if acao in acoes:
            print("{} para {}".format(local, acao))
            break

displacy.render(documento, style='dep', jupyter=True, options={'distance':90})




p1 = pln('olá')
p2 = pln('oi')
p3 = pln('ou')

p1.similarity(p2)
p2.similarity(p1)
p1.similarity(p3)
p2.similarity(p3)


texto1 = pln('Quando será lançado o novo filme?')
texto2 = pln('O novo filme será lançado mês que vem')
texto3 = pln('Qual a cor do carro?')

texto1.similarity(texto2)
texto1.similarity(texto3)


texto = pln('gato cachorro cavalo pessoa')
for texto1 in texto:
    #print('-----', texto1)
    for texto2 in texto:
        #print(texto2)
        similaridade = int(texto1.similarity(texto2) * 100)
        print("{} é {} similar a {}".format(texto1, similaridade, texto2))



documento = pln('Estou aprendendo processamento de linguagem natural, curso em Curitiba')
documento1 = 'Estou aprendendo processamento de linguagem natural, curso em Curitiba'
documento1.split(' ')
for token in documento:
    print(token)






