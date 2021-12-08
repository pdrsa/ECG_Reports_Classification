#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Importando pacotes
import multiprocessing
from unidecode import unidecode
from joblib import Parallel, delayed
import numpy as np
import string
import pandas as pd
import time
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process
from fuzzyJoaoAdaptado import partial_ratio
from argparse import ArgumentParser


# In[2]:


parser = ArgumentParser(description="Classficador ECG")

parser.add_argument('dataset', help = "O nome do dataset que será usado.")

args = parser.parse_args()


# In[2]:


#Importando dados
start_time      = time.time()
print("Lendo dados...\n")
laudos_completo = pd.read_csv(args.dataset, sep = ';')
dicionario      = pd.read_csv("../data/DicionarioECG_semSiglas.csv")


# In[3]:


#Função que recebe o dicionário e uma String e retorna uma lista com a similaridade de cada doença à String
def checarSimilaridadeDoencas(texto, dicionario):
    listaScore = []
    for row in dicionario.itertuples():
        maiorScore = 0
        maiorScoreL = []
        for diag in row:
            if ((type(diag) is not int and type(diag) is not float) and (type(texto) is not float)):
                texto = texto.translate(str.maketrans('', '', string.punctuation))
                texto = unidecode(texto)
                score = partial_ratio(diag.lower(), texto.lower())
                if score[2] > maiorScore: 
                    maiorScore = score[2]
                    maiorScoreL = score
        if (maiorScore > 0):
            listaScore.append(maiorScoreL)
        else:
            listaScore.append(['noMatch', 'noMatch', 0])
    return listaScore


# In[4]:


print("Trabalhando dados...\n")
idExames   = laudos_completo['ID_EXAME']
idLaudos   = laudos_completo['ID_LAUDO']
textos     = laudos_completo['CONTEUDO']


# In[11]:


#Preenchendo o DataFrame

start_time = time.time()

num_cores = multiprocessing.cpu_count()

print("Início da parte demorada do código...\n")
print(time.time() - start_time, "segundos já se passaram...\n")
resultado = Parallel(n_jobs = num_cores)(delayed(checarSimilaridadeDoencas)
                              (laudos_completo['CONTEUDO'][i], dicionario)
                    for i in range(len(laudos_completo)))
                              
print("Fim da parte demorada...\n")
print((time.time() - start_time)/3600, "horas desde o início do código...\n")

# for i in range(100):
#     df.loc[i,"idExame"]           = laudos_completo.loc[i,"ID_EXAME"]
#     df.loc[i,"idLaudo"]           = laudos_completo.loc[i,"ID_LAUDO"]
#     df.loc[i,"texto"]             = laudos_completo.loc[i,"CONTEUDO"]
#     listaSimilaridade             = checarSimilaridadeDoencas\
#                                     (laudos_completo.loc[i,"CONTEUDO"], dicionario)
#     scores = [x[2] for x in listaSimilaridade]
#     posicoes = [[x[0],x[1]] for x in listaSimilaridade]
#     df.loc[i,"scorePatologias"]   = scores
#     df.loc[i,"posicaoString"]     = posicoes


# In[53]:


header = "idExame; idLaudo; texto; resultado"


# In[35]:


#LEMBRE-SE DE TIRAR A PONTUAÇÃO QUANDO FOR OLHAR OS RESULTADOS


# In[54]:


print("Escrevendo dados...\n")
nomeArquivo = (args.dataset + "_result")
arquivo = open(nomeArquivo, 'w')
# arquivo.write(header)
# arquivo.write("\n")
for i in range(len(laudos_completo)):
        arquivo.write(str(idExames[i]))
        arquivo.write(';')
        arquivo.write(str(idLaudos[i]))
        arquivo.write(';')
        arquivo.write(str(str(textos[i]).translate(str.maketrans('', '', string.punctuation))))
        arquivo.write(';')
        arquivo.write(str(resultado[i]))
        arquivo.write("\n")
arquivo.close()
print("Concluído!")

