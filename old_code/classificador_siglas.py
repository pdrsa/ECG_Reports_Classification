#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importando pacotes
import numpy as np
import pandas as pd
import spacy
from argparse import ArgumentParser


# In[3]:


parser = ArgumentParser(description="Classficador ECG")

parser.add_argument('dict', help = "O nome do dicionario que será usado.")
parser.add_argument('dataset', help = "O nome do dataset que será usado.")
parser.add_argument('-csv', action = 'store', help = "O prefixo do nome do CSV em que você deseja guardar os dados processados.", default="result")

args = parser.parse_args()


# In[2]:


#Importando dados
print("Novo processo!")
print("Lendo dados...")
laudos_completo = pd.read_csv((args.dataset), sep = ";")
dicionario = pd.read_csv((args.dict))


# In[3]:


#Função que recebe o dicionário e uma String e retorna uma lista com a similaridade de cada doença à String
def checarSimilaridadeDoencas(texto, dicionario):
    listaScore = []
    for row in dicionario.itertuples():
        match = False
        for diag in row:
            if ((type(diag) is not int and type(diag) is not float) and (type(texto) is not float)):
                while(texto.find(diag) != -1):
                    pos = texto.find(diag)
                    num1 = 32
                    num2 = 32
                    #Numero dos caracteres na tabela ASCII.
                    if (pos > 0):
                        num1  = ord(texto[pos-1])
                    if (pos+len(diag)+1 < len(texto)):
                        num2  = ord(texto[pos + len(diag) + 1])

                    #Condições checam se os caracteres são pontuações ou espaço.
                    cond1 = (num1 >= 32 and num1 <= 47) or                    (num1 == 61 or num1 == 72 or num1 == 58                    or num1 == 59 or num1 == 95)

                    cond2 = (num2 >= 32 and num2 <= 47) or                    (num2 == 61 or num2 == 72 or num2 == 58                    or num2 == 59 or num2 == 95)


                    if(not cond1 or not cond2):
                            texto = texto[:pos] + texto[pos+len(diag)+1:]
                            continue
                    listaScore.append([pos, pos+len(diag)+1, 100])
                    match = True
                    break
                if(match):
                    break
        if(not match):
            listaScore.append(['noMatch', 'noMatch', 0])
    return listaScore


# In[4]:


dfHeader          = "idExame,idLaudo; texto; scorePatologias; posicaoString" 
idExames          = laudos_completo["ID_EXAME"]
idLaudo           = laudos_completo["ID_LAUDO"]
texto             = laudos_completo["CONTEUDO"]
print("Processando dados...")
listaSimilaridade =[checarSimilaridadeDoencas(texto, dicionario) for texto in laudos_completo["CONTEUDO"]]
print("Selecionando dados...")
scores            = [[x[2] for x in y] for y in listaSimilaridade]
print("Ainda selecionando...")
posicoes          = [[[x[0],x[1]] for x in y] for y in listaSimilaridade]
print("Gravando dados na memoria...")

# In[5]:


nomeArquivo = ((args.dataset) + "result")
arquivo = open(nomeArquivo, 'w')
arquivo.write(dfHeader)
arquivo.write("\n")
for i in range(len(scores)):
    arquivo.write(str(idExames[i]))
    arquivo.write(';')
    arquivo.write(str(idLaudo[i]))
    arquivo.write(';')
    arquivo.write(str(texto[i]))
    arquivo.write(';')
    arquivo.write(str(scores[i]))
    arquivo.write(';')
    arquivo.write(str(posicoes[i]))
    arquivo.write("\n")
arquivo.close()


# In[77]:


#Preenchendo o DataFrame
# for i in range(len(laudos_completo)):
#     df.loc[i,"idExame"]           = laudos_completo.loc[i,"ID_EXAME"]
#     df.loc[i,"idLaudo"]           = laudos_completo.loc[i,"ID_LAUDO"]
#     df.loc[i,"texto"]             = laudos_completo.loc[i,"CONTEUDO"]
#     listaSimilaridade             = checarSimilaridadeDoencas\
#                                     (laudos_completo.loc[i,"CONTEUDO"], dicionario)
#     scores = [x[2] for x in listaSimilaridade]
#     posicoes = [[x[0],x[1]] for x in listaSimilaridade]
#     df.loc[i,"scorePatologias"]   = scores
#     df.loc[i,"posicaoString"]     = posicoes

