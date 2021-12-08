#!/usr/bin/env python
# coding: utf-8

# In[257]:


print("Importing packages...")
import pandas as pd
import numpy as np
from bisect import bisect_left


# In[269]:


def BinarySearch(a, x): 
    i = bisect_left(a, x) 
    if i != len(a) and a[i] == x: 
        return True 
    else: 
        return False


# In[2]:


print("Reading data...")
passingScores = pd.read_csv("../../../data/resultados/sem_siglas/nota_de_corte_manual")
#WA stands for Without Acronyms and OA stands for Only Acronyms 
scoresOA = pd.read_csv("../../../data/resultados/apenas_siglas/scores_apenas_siglas.csv")
scoresWA = pd.read_csv("../../../data/resultados/sem_siglas/resultado_sem_siglas_scores.csv")


# In[303]:


print("Processing data [1/2]...")
#Only acronyms green zone
positivesOA = {row[1]: sorted(list(scoresOA["idExame"][scoresOA[row[1]] == 100])) for row in passingScores.itertuples()}

#Without acronyms green zone
#row[1] is the diagnosis name and row[2] is the passing score for that diagnosis
positivesWA = {row[1]: sorted(list(scoresWA["idExame"][scoresWA[row[1]] >= row[2]])) for row in passingScores.itertuples()}


# In[368]:


print("Processing data [2/2]...")
#Creating the label for all lines of the dataset
#line[1] equals the idExame of that line
labels = []
i = 0

for line in scoresOA.itertuples():
    actualLabel = []
    actual = {}
    actual["idExame"] = line[1]
    for row in passingScores.itertuples():
        condition1 = BinarySearch(positivesOA[row[1]], line[1])
        condition2 = BinarySearch(positivesWA[row[1]], line[1])
        if condition1 or condition2:
            actualLabel.append(True)
        else:
            actualLabel.append(False)
    actual["label"] = actualLabel
    labels.append(actual)
    
    if(i % 25000 == 0):
        print ("[", i, "/2497000] done...")
    i = i+1


# In[367]:


#Creating two dataframes, one of which will have the idExame of all the lines that are in the green zone
dfLabels     = pd.DataFrame(labels)
greenZoneIds = [dfLabels["idExame"][i] for i in range(len(dfLabels)) if np.any(dfLabels["label"][i])]
greenZoneIds = pd.DataFrame(greenZoneIds, columns = ["id"])
print("Saving data...")
dfLabels.to_csv("../../../data/resultados/resultLabels.csv")
greenZoneIds.to_csv("../../../data/resultados/greenZoneIds.csv")
print("Concluido!")
