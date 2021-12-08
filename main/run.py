#!/usr/bin/env python
# coding: utf-8

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--skipscore", help="Skip the score determination step of the process.", action="store_true")
parser.add_argument("--scoreonly", help="Will make only the score determination step of the process.", action="store_true")
parser.add_argument("--makethresh", help="Will make the thresholding proccess for the scores.", action="store_true")
parser.add_argument("--loadY", help="Will load Y from output", action="store_true")
args = parser.parse_args()


if(args.skipscore):
    print("You have choosen to skip the score determination part of the process. Be aware that the file in the folder output will be used instead.")
if(args.scoreonly):
    print("Be aware that only the score determination part of the process will be made.")
if(args.makethresh):
    print("You have choosen to make the thresholding of the scores. This will create a new binary array from the scores.")
if(args.loadY):
    print("You have choosen to load the Y_train.")
# In[95]:


print("Loading modules, classes and methods...")
import multiprocessing
from unidecode import unidecode
from joblib import Parallel, delayed
import numpy as np
import string
import pandas as pd
import time
import os
import numpy as np
import re
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchvision
import pickle
from unicodedata import normalize
from keras.preprocessing.text import Tokenizer
from torchvision import transforms, datasets
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
import textclassification as tc
from textclassification import Net
import math

# In[348]:


#GLOBAL VARIABLES
N_CORES = multiprocessing.cpu_count()

# Dataset
N_CLASS          = 35

#Scores
STARTING         = 0
ENDING           = 2000000
VAL              = 100000
BATCH_SCORES     = 1000

#Neural Network
TOP_WORDS       = 10000
REBUILD_DATA    = True
BIDIRECTIONAL   = True
EPOCHS          = 20
BATCH_SIZE      = 256


# In[46]:
print("Loading data...")

valid = np.load("../light_data/new_classes.npy")
prevalences = pd.read_csv("../light_data/prevalencia_e_scores_com_siglas.csv")
dicionarioCsv = pd.read_csv("../light_data/DicionarioECG_completo.csv", index_col = 0)
db = pd.read_csv("../../data/DATA_LAUDOS_TEXTO_formato1", sep = ";")


print("Cutting slice. From {} to {}...".format(STARTING, ENDING))
db = db[STARTING:ENDING]
db = db.reset_index()


# In[96]:


print("Preparing texts...")
texts = [tc.clean_text(text) for text in db["CONTEUDO"]]
    
dicionario = {}
for row in dicionarioCsv.itertuples():
    aux = []
    for diag in row:
        if type(diag) is str: aux.append(tc.clean_text(diag))
    dicionario[row[0]] = aux


# In[205]:

if (not args.skipscore and not args.loadY):
    print("Calculating scores...\n")

    def return_scores(text, dicionario):
        scores = [max([tc.make_score(text, diag) for diag in dicionario[i]]) if valid[i][1] else 0 for i in range(74)]
        return scores

    batch = BATCH_SCORES
    print("Working in batches of", batch)
    with open('output/score_values.csv', 'w') as f:
        f.write("id_exame, scorings\n")
        #     errors = []
    startTime = time.time()
    for i in range(0, len(db), batch):
        print(i,"/",len(db))
        startBatch = time.time()
        #try:
    #     scores = [return_scores(text, dicionario) for text in texts[i:i+batch]]
        scores = Parallel(n_jobs = N_CORES)(delayed(return_scores)
                                  (text, dicionario)
                        for text in texts[i:i+batch])

        if(i == 0): result = np.array(scores)
        else: result = np.concatenate((result, np.array(scores)), axis = 0)
        with open('output/score_values.csv', 'a') as f:
            for j in range(i,i+batch):
                f.write(str(db["ID_EXAME"][j]))
                f.write(',"')
                f.write(str(scores[j-i]))
                f.write('"\n')
    #     except:
    #         print("ERROR!!!!!")
    #         errors.append([i, i+batch])
    #     errors = np.array(errors)
    #     np.save("errors.npy", errors)

        expectedTime = (((time.time() - startTime)/(i+batch)) * (len(db))) - (time.time() - startTime)
        timeBatch    = time.time() - startBatch
        print("This batch has been done in", int(timeBatch/60), "minutes and", timeBatch%60,"seconds!")
        print("Expected time for ending is around", int(expectedTime/3600), "hours and ", int((expectedTime%3600)/60),"minutes!")
    #         with open('../../data/resultados/scorings1.csv', 'a') as f:
    #             for j in range(i,i+batch):
    #                 f.write(str(db["ID_EXAME"][j]))
    #                 f.write(',"')
    #                 f.write(str(scores[j-i]))
    #                 f.write('"\n')
    print("Y of training data defined!!! Saving...")
    np.save("output/score_values.npy", result)
    print("Saved!")
else:
    result = np.load("output/score_values.npy")

if(args.scoreonly):
    print("Calculating thresholds...")
    scores_thresh = []
    nClass = 0
    for i in range(len(result[0])):
        if(not valid[i][1]): continue
        print("Calculating for", prevalences.loc[i]["Diagnostico"])
        temp_result       = result[:,i]
        temp_result = np.sort(temp_result, kind = 'mergesort')
        ocurrences     = len(result) * prevalences.loc[i]["Prevalencia"]
        ocurrences     = math.ceil(ocurrences)
        threshAt       = temp_result[-ocurrences]
        
        scores_thresh.append(threshAt)
        print("Threshold by Prevalence =",temp_result[-ocurrences])
        print("Threshold added =", threshAt)
        nClass += 1    
    np.save("output/score_thresholds.npy", scores_thresh)
    import sys
    exit()
        
    
# In[284]:
## This is especifically for the ECG dataset. We are maping the 74 old classes to the 35 desired.
## If you make trans[i] = i, for all classes, it should work for any dataset.
# trans = {}
# for i in range(N_CLASS):
#     trains[i] = i

print("Translating from old to new...")
trans = {}
for i in range(N_CLASS):
    aux = 0
    for z in range(74):
        aux += valid[z][1]
        if(aux == (i+1)): 
            trans[i] = z
            break
    print(i,"->",trans[i])

if(args.makethresh):
    print("Calculating thresholds and creating binary arrays...")
    scores_thresh = []
    y_bin = np.zeros((len(result),valid[:,1].sum()))
    nClass = 0
    for i in range(len(result[0])):
        if(not valid[i][1]): continue
        print("Calculating for", prevalences.loc[i]["Diagnostico"])
        temp_result       = result[:,i]
        temp_result = np.sort(temp_result, kind = 'mergesort')
        ocurrences     = len(result) * prevalences.loc[i]["Prevalencia"]
        ocurrences     = math.ceil(ocurrences)
        threshAt       = temp_result[-ocurrences]
        for diag in dicionario[i]:
            if(diag[-1] == '#'):
                threshAt       = max([threshAt, 60])
                break
        scores_thresh.append(threshAt)
        for j in range(len(result)):
            if result[j][i] >= scores_thresh[-1]: y_bin[j][nClass] = 1
        print("Threshold by Prevalence =",temp_result[-ocurrences])
        print("Threshold added =", threshAt)
        nClass += 1
        
elif(not args.loadY):
    y_bin = np.zeros((len(result),valid[:,1].sum()))
    scores_thresh = np.load("output/scores_thresholds.npy")
    print("Thresholds:")
    print()
    for i in range(len(scores_thresh)):
        print(dicionario[trans[i]][0],"->",scores_thresh[i])
    for j in range(len(result)):
        for i in range(N_CLASS):
            if result[j][trans[i]] >= scores_thresh[i]: y_bin[j][i] = 1
                
    print("Did it! Some examples:")
    for i in range(N_CLASS):
        print("Example of", dicionario[trans[i]][0])
        print()
        for l in range(len(texts)):
            if y_bin[l][i]:
                print(texts[l])
                break
        print()
        print()

    np.save("output/score_bin.npy", y_bin)

if(args.loadY):
    y_bin = np.load("output/score_bin.npy")


# In[297]:

print("Calculating the greenzone...")
gZoneIdx = []
gZoneIds = []
gZoneLen = 0
removeIdx = []
for i in range(len(result)):
    if y_bin[i].sum() > 0:
        gZoneIds.append(db["ID_EXAME"][i])
        gZoneIdx.append(i)
        gZoneLen += 1
    else:
        removeIdx.append(i)
        
np.save("output/gZoneIds.npy", gZoneIds)
np.save("output/gZoneIdx.npy", gZoneIdx)
print("Done! GreenZone has", gZoneLen, "registers!")
print("Removing useless entries...")
y_bin = np.delete(y_bin, removeIdx, axis = 0)
texts = np.delete(np.array(texts), removeIdx)
print("Done!", len(texts), "texts and",len(y_bin),"labels left!")


# In[305]:


print("Creating neural network...")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on a GPU :D")
else:
    device = torch.device("cpu")
    print("Running on a CPU :/")


# In[347]:

print("Creating vocabulary...")
tokenizer = Tokenizer(num_words = TOP_WORDS, split = ' ')
tokenizer.fit_on_texts(texts)
print("Tokenizing texts...")
train_X   = tokenizer.texts_to_sequences(texts)
with open("output/tokenizer_"+str(TOP_WORDS)+".pickle", 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("X of training data defined!!! Saving...")
np.save("output/x_train.npy", train_X)

# In[324]:



print("Reshaping training data...")
train_X  = [torch.Tensor(i).type(torch.LongTensor) for i in train_X]
train_X  = pad_sequence(train_X, batch_first=True).type(torch.LongTensor)
train_y  = torch.Tensor(y_bin)
seq_size = len(train_X[0])
print("Training input has",seq_size,"dimensions!")

    
print("Defining validation...")
val_X     = train_X[VAL:]
train_X   = train_X[:VAL]
val_y     = train_y[VAL:]
train_y   = train_y[:VAL]


# In[346]:

net = Net(seq_size, TOP_WORDS).to(device)
optimizer = optim.Adam(net.parameters(), lr = 0.0005)
loss_function = nn.BCELoss()
# Training the model
print("\nAlexa, play eye of the tiger. It's training time!\n\n")
bestLoss = 1
for epoch in range(EPOCHS):
    print("Epoch", epoch+1)
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
#         print(i, i+BATCH_SIZE)
        batch_X = train_X[i:i+BATCH_SIZE]
        batch_y = train_y.squeeze()[i:i+BATCH_SIZE]

        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        net.zero_grad()
        outputs = net(batch_X)
        loss    = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    actualLoss = loss.data.tolist()
    print("Loss:", actualLoss,"\n")
    if(actualLoss > 1):
        print("Bad result! Stopping...")
        break
    if(actualLoss < bestLoss):
        print("This is the best loss until now! Saving...")
        bestLoss = actualLoss
        torch.save(net.state_dict(), "output/network_model.pth")
    else: print("Bad candidate. Proceeding to next.")

print("Finished training!")

with open('output/parameters', 'w') as p:
    p.write(str(seq_size) + "\n" + str(TOP_WORDS))

print("Validation...")

finalResult = torch.Tensor()
size = 64
for i in tqdm(range(0, val_X.size()[0], size)):
    with torch.no_grad():
        result      = net(val_X[i: min(i+size, val_X.size()[0])])
    finalResult = torch.cat((finalResult, result), 0)
    
val_score = np.array(finalResult)
val_y     = np.array(val_y)


print("Calculating best candidates for threshold...")

n_class = N_CLASS
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
thresholds = dict()
roc_auc = dict()
for i in range(n_class):
    fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(val_y[:, i], val_score[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(val_y.ravel(), val_score.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

print("Calculating thresholds...")
#Calculate limits by maximizing F1
limits = []
for j in tqdm(range(n_class)):
    bigf1 = 0
    for threshold in thresholds[j]:
        y_bin = []
        for row in val_y[:,j]:
            if row > thresholds[i]:
                y_bin.append(1)
            else:
                y_bin.append(0)
        y_bin = np.array(y_bin)
        precision, _, f1, _ = precision_recall_fscore_support(val_y[:,j], y_bin, average = 'binary')
        
        if(f1 > bigf1 and precision > 0):
            bigf1 = f1
            maxi = threshold
    limits.append(maxi)

np.save("output/nn_thresholds.npy", limits)

print("\nCompleted!\nYou can find all the files that this script has generated in the output folder."
print("Thresholds for testing are 'output/nn_thresholds'. The network state is 'output/network_model.pth'")
