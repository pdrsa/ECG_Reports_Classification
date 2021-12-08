#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math


# In[2]:


tamanhoDB    = 2427689
listaScores  = []

# In[1]:


dfPrevalencias = pd.read_csv("../data/prevalencia_e_scores_com_siglas.csv", index_col=0)
resultados = pd.read_csv("../data/resultados/sem_siglas/resultado_sem_siglas_scores.csv")
diagnosticos = dfPrevalencias["Diagnostico"]

# In[36]:


print("Started...")
for i in range(len((dfPrevalencias))):
    listaResultados       = resultados[dfPrevalencias.loc[i+1]["Diagnostico"]]
    listaResultados       = listaResultados.to_list()

    listaResultados       = [int(score) for score in listaResultados]

    listaResultados.sort(reverse=True)

    numeroOcorrencias     = tamanhoDB * dfPrevalencias.loc[i+1]["Prevalencia"]
    numeroOcorrencias     = math.ceil(numeroOcorrencias)
    listaScores.append(listaResultados[numeroOcorrencias])


# In[45]:


arquivo = open('../data/resultado_sem_siglas/nota_de_corte_estimada', 'w')
arquivo.write("diagnostico, score")
arquivo.write("\n")
for i in range(len(scores)):
    arquivo.write(str(diagnosticos[i]))
    arquivo.write(',')
    arquivo.write(str(listaScores[i]))
    arquivo.write("\n")
arquivo.close()
