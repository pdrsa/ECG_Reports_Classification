#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import re
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pickle
from torch.nn.utils.rnn import pad_sequence
from unicodedata import normalize
from keras.preprocessing.text import Tokenizer
from torchvision import transforms, datasets
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt


# In[20]:


# Global variables
REBUILD_DATA = False
BIDIRECTIONAL = True
TOP_WORDS = 10000
EPOCHS = 30
BATCH_SIZE = 256


# In[4]:


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on a GPU :D")
else:
    device = torch.device("cpu")
    print("Running on a CPU :/")


# In[5]:


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# In[22]:


# Oi, Derick
# To comentando como diálogo porque acho mais fácil explicar o código assim. Se o comentário estiver errado é pq essa é a parte que eu entendi errado.
class Net(nn.Module):
    def __init__(self, seq_size):
        
        print("Building NN...")
        embedding_dim = 128
        lstm_out_dim = 128
        num_embeddings = TOP_WORDS
        num_of_classes = 35
        
        super().__init__()
        #Camada de Embedding, o padding_idx é um argumento que eu descobri que é usada para falar para a camada que os números no fim de cada vetor são apenas lixo
        self.l1 = nn.Embedding(num_embeddings, embedding_dim, padding_idx = 0)
        #Eu não entendo muito bem o que essa camada faz. Pelo que eu entendi é algo probabilístico. Mas ela n altera o shape.
#         self.l2 = nn.Dropout(p=0.4)
        #A LSTM recebe os Embeddings e cospe o mesmo número de vetores que eu passei para ela. Não sei se eu deveria alterar o número de camadas da LSTM.
        #Se usar menos de 2 não dá pra colocar Dropout pq o Dropout é aplicado em todas as camadas menos na última.
        self.l3 = nn.LSTM(embedding_dim, lstm_out_dim, dropout = 0.2, num_layers = 2, bidirectional = BIDIRECTIONAL)
        #É o seguinte. Como as dimensões de entrada são estáticas, eu adicionei elas manualmente na camada linear para conseguir fazer o flatten.
        self.l4 = nn.Flatten()
        #Dimensao do vetor de entrada X dimensao da lstm
        self.l5 = nn.Linear(seq_size * lstm_out_dim * (2 if BIDIRECTIONAL else 1), num_of_classes)
        
    
    def forward(self, x):
        #Aqui eu só to passando o input pelas camadas mesmo
        x    = self.l1(x)
#         x    = self.l2(x)
        #A camada de LSTM retorna uma tupla, o vetor que eu quero é a primeira posição da tupla, por isso recebo assim.
        #Acho que a segunda camada da LSTM só é util ao passar de uma camada da LSTM para a outra.
        x, _ = self.l3(x)
        x    = self.l4(x)
        x    = self.l5(x)
        #Aqui eu aplico o softmax. Especifico o número de dimensões para ser um e tal. Não sei o que não está funcionando :c.
        x    = F.softmax(x, dim = 1)
            
        return x                    


# In[7]:


def clean_text(x):
    if type(x) is str:
        pattern = r'[^a-zA-z0-9!:.,?\s]'
        x = normalize('NFKD', x).encode('ASCII', 'ignore').decode('ASCII')
        x = re.sub(pattern, '', x)
        return x
    else:
        return ""


# In[8]:


def load_data():
    print("Loading data:\n")
    print("Ids...")
    greenZoneIdx = np.load('../../light_data/greenZoneIndex.npy')
    print("Text...")
    db           = pd.read_csv('../../../data/DATA_LAUDOS_TEXTO_formato1', sep = ";")
    print("Labels...")
    resultLabels = np.load('../../../data/resultados/scores/allLabels.npy')
    print("Data loaded!\n")
    
    print("Preprocessing data:\n")
    print("Selecting slices...")
    text_data      = db[db.index.isin(greenZoneIdx)]
    labels   = [resultLabels[i] for i in greenZoneIdx]
    # dados_validacao = dados_texto[dados_texto['ID_EXAME'].isin(ids_achados['id'][-400000:])]

    text   = text_data['CONTEUDO']
    text   = text[:-300000]
    labels = labels[:-300000]
    
    return text, labels


# In[9]:


def tokenize(text, tokenizer, fit = False):
    print("Tokenizing...")
    # Creating vocabulary
    if fit:
        tokenizer.fit_on_texts(text)
    # Vectorizing text
    train_X   = tokenizer.texts_to_sequences(text)
    # Saving tokenizer
    with open("../../light_data/pytorch_tokenizer.pickle", 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return train_X


# In[10]:


if REBUILD_DATA:
    text, labels = load_data()
    text         = [clean_text(i) for i in text]
    tokenizer = Tokenizer(num_words = TOP_WORDS, split = ' ')
    train_X   = tokenize(text, tokenizer, fit = True)
    np.save("../../../data/training_data/training_data_X.npy", train_X)
else:
    labels       = np.load('../../../data/resultados/scores/allLabels.npy')
    greenZoneIdx = np.load('../../light_data/greenZoneIndex.npy')
    labels       = [labels[i] for i in greenZoneIdx]
    train_X      = np.load('../../../data/training_data/training_data_X.npy', allow_pickle = True)
    labels       = labels[:len(train_X)]


# In[11]:


train_y  = labels
print("Transforming lists into tensors...")
train_X  = [torch.Tensor(i).type(torch.LongTensor) for i in train_X]
train_X  = pad_sequence(train_X, batch_first=True).type(torch.LongTensor)
train_y  = torch.Tensor(train_y)
seq_size = len(train_X[0])
print("Data preprocessed!\n")
print("Number of input dimensions: ", seq_size)


# In[23]:


net = Net(seq_size).to(device)
optimizer = optim.Adam(net.parameters(), lr = 0.0005)
loss_function = nn.BCELoss()
# Training the model
print("Okay, here we go.\nWe have",len(train_X),"tokenized texts and",len(train_y),"labels to train.")
print("\nAlexa, play eye of the tiger. It's train time!\n\n")
for epoch in range(EPOCHS):
    print("Epoch ", epoch)
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

    print("End of Epoch ", epoch, "!\n Loss: ", loss,"\n")

print("Finished training! Loss: ", loss)
print("\nSaving model...")
torch.save(net.state_dict(), "../../../data/trained_models/pytorch_checkpoint_3.pth")
print("Finish!")

