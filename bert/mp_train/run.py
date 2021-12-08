#!/usr/bin/env python
# coding: utf-8

# In[16]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import transformers
from joblib import Parallel, delayed
import multiprocessing
import time
from model_mp import marcapassomodel


# In[17]:


print("O Modelo: \n\n")
model = marcapassomodel()
print(model.summary())


# In[18]:


print("Importando os dados...")
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test  = np.load("x_test.npy")
y_test  = np.load("y_test.npy")


# In[19]:


#Okay, training
print("Treinando...\n\n")
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=32, epochs=10, validation_split=0.1,
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='max', min_delta=1),
             tf.keras.callbacks.ModelCheckpoint(filepath='best_model_mp', monitor='val_accuracy', save_best_only=True)]
)
model.save("last_model")


# In[20]:


ordem_classes = ['chagas', 'miocardiopatia isquêmica', 'cardiopatia valvar',
       'cardiomiopatia hipertrófica', 'cardiopatia congênita',
       'síndrome do QT longo', 'síndrome de Brugada',
       'fibrilação ventricular idiopática',
       'displasia arritmogênica do VD', 'miocardiopatia idiopática']


# In[22]:


print("Prevendo com o último modelo...")
y_score = model.predict(x_test, batch_size = 32)
np.save("predict_tf_last_mp.npy", y_score)

print("Prevendo com o melhor modelo...")
# model = tf.keras.models.load_model('best_model')
y_score = model.predict(x_test, batch_size = 32)
np.save("predict_tf_best_mp.npy", y_score)


# In[26]:


print("Testando melhor modelo...")

n_class = 10
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
thresholds = dict()
roc_auc = dict()
for i in range(n_class):
    fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

limits = []
for j in range(n_class):
    bigf1 = 0
    for threshold in thresholds[j]:
        y_bin = []
        for row in y_score[:,j]:
            if row > threshold:
                y_bin.append(1)
            else:
                y_bin.append(0)
        y_bin = np.array(y_bin)
        precision, _, f1, _ = precision_recall_fscore_support(y_test[:,j], y_bin, average = 'binary')
        
        if(f1 > bigf1 and precision > 0):
            bigf1 = f1
            maxi = threshold
    limits.append(maxi)
    
def getMetrics(y_test, y_score, thresholds):
    
    #First we turn into binary
    y_bin = []
    for j in range(len(y_score)):
        ans = []
        for i in range(n_class):
            if y_score[j][i] >= thresholds[i]:
#             if y_label[j][i]:
                ans.append(1)
            else:
                ans.append(0)
        y_bin.append(np.array(ans))
    y_bin = np.array(y_bin)
    np.save("bin_tf_best.npy", y_bin)
    
    #Then we calculate
    target_names = ["(" + ordem_classes[i] + ") Class" + str(i) for i in range(n_class)]
    precision = dict()
    recall = dict()
    f1 = dict()
    sup = dict()
    for i in range(n_class):
        precision[i], recall[i], f1[i], sup[i] = precision_recall_fscore_support(y_test[:,i], y_bin[:,i], average = 'binary')
    return precision, recall, f1, sup

precision, recall, f1, _ = getMetrics(y_test, y_score, limits)
f1 = f1.items()
df = pd.DataFrame(columns = ["Class", "Precision", "Recall", "F1", "Ocurrences"])
for row in f1:
    n = row[0]
    sup = y_test[:,n].sum()
    new_row = {'Class': str(ordem_classes[n]), 'Precision': precision[n], 'Recall': recall[n], 'F1': row[1], "Ocurrences": sup}
    df = df.append(new_row, ignore_index = True)
df = df.set_index("Class")
df.to_csv("resultBestModelMP.csv")


# In[27]:


print("Testando último modelo...")

y_score = np.load("predict_tf_last_mp.npy")

n_class = 10
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
thresholds = dict()
roc_auc = dict()
for i in range(n_class):
    fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

limits = []
for j in range(n_class):
    bigf1 = 0
    for threshold in thresholds[j]:
        y_bin = []
        for row in y_score[:,j]:
            if row > threshold:
                y_bin.append(1)
            else:
                y_bin.append(0)
        y_bin = np.array(y_bin)
        precision, _, f1, _ = precision_recall_fscore_support(y_test[:,j], y_bin, average = 'binary')
        
        if(f1 > bigf1 and precision > 0):
            bigf1 = f1
            maxi = threshold
    limits.append(maxi)
    
def getMetrics(y_test, y_score, thresholds):
    
    #First we turn into binary
    y_bin = []
    for j in range(len(y_score)):
        ans = []
        for i in range(n_class):
            if y_score[j][i] >= thresholds[i]:
#             if y_label[j][i]:
                ans.append(1)
            else:
                ans.append(0)
        y_bin.append(np.array(ans))
    y_bin = np.array(y_bin)
    np.save("bin_tf_last.npy", y_bin)
    
    #Then we calculate
    target_names = ["(" + ordem_classes[i] + ") Class" + str(i) for i in range(n_class)]
    precision = dict()
    recall = dict()
    f1 = dict()
    sup = dict()
    for i in range(n_class):
        precision[i], recall[i], f1[i], sup[i] = precision_recall_fscore_support(y_test[:,i], y_bin[:,i], average = 'binary')
    return precision, recall, f1, sup

precision, recall, f1, _ = getMetrics(y_test, y_score, limits)
f1 = f1.items()
df = pd.DataFrame(columns = ["Class", "Precision", "Recall", "F1", "Ocurrences"])
for row in f1:
    n = row[0]
    sup = y_test[:,n].sum()
    new_row = {'Class': str(ordem_classes[n]), 'Precision': precision[n], 'Recall': recall[n], 'F1': row[1], "Ocurrences": sup}
    df = df.append(new_row, ignore_index = True)
df = df.set_index("Class")
df.to_csv("resultLastModelMP.csv")


# In[28]:


print("Finalizado!")

