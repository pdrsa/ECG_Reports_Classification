#!/usr/bin/env python
# coding: utf-8

# In[34]:


#Importando pacotes
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process
from fuzzyJoao import partial_ratio
from argparse import ArgumentParser


# In[82]:


parser = ArgumentParser(description="Classficador ECG")

parser.add_argument('dict', help = "O nome do dicionario que será usado.")
parser.add_argument('dataset', help = "O nome do dataset que será usado.")
parser.add_argument('-csv', action = 'store', help = "O prefixo do nome do CSV em que você deseja guardar os dados processados.", default="result")

args = parser.parse_args()


# In[2]:


#Importando dados
laudos_completo = pd.read_csv((args.dataset), sep = ";")
dicionario = pd.read_csv((args.dict), sep = ";")

# In[1]:


#Função que recebe o dicionário e uma String e retorna uma lista com a similaridade de cada doença à String
def checarSimilaridadeDoencas(texto, dicionario):
    listaScore = []
    for row in dicionario.itertuples():
        maiorScore = 0
        maiorScoreL = []
        for diag in row:
            if ((type(diag) is not int and type(diag) is not float) and (type(texto) is not float)):
                score = partial_ratio(diag.lower(), texto.lower())
                if score[2] > maiorScore: 
                    maiorScore = score[2]
                    maiorScoreL = score
        if (maiorScore > 0):
            listaScore.append(maiorScoreL)
        else:
            listaScore.append(['noMatch', 'noMatch', 0])
    return listaScore


# In[38]:


#Criando o DataFrame
df = pd.DataFrame(columns=['idExame','idLaudo','texto','scorePatologias','posicaoString'])


# In[77]:


#Preenchendo o DataFrame
for i in range(len(laudos_completo)):
    df.loc[i,"idExame"]           = laudos_completo.loc[i,"ID_EXAME"]
    df.loc[i,"idLaudo"]           = laudos_completo.loc[i,"ID_LAUDO"]
    df.loc[i,"texto"]             = laudos_completo.loc[i,"CONTEUDO"]
    listaSimilaridade             = checarSimilaridadeDoencas                                    (laudos_completo.loc[i,"CONTEUDO"], dicionario)
    scores = [x[2] for x in listaSimilaridade]
    posicoes = [[x[0],x[1]] for x in listaSimilaridade]
    df.loc[i,"scorePatologias"]   = scores
    df.loc[i,"posicaoString"]     = posicoes


# In[78]:


df.to_csv(((args.dataset) + "_" + (args.csv)))

