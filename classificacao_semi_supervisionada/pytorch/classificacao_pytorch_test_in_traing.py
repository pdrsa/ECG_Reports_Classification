#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
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
import keras
from torchvision import transforms, datasets
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import pylab as pl
import matplotlib.pyplot as plt


# In[2]:


corte_classes = [84, 85, 96, 84, 97, 94, 81, 100, 83, 95, 83, 83, 81, 92, 95, 81, 85, 92, 84, 83, 86, 89, 83, 92, 93, 92, 83, 83, 87, 
                100, 92, 80, 84, 87, 91, 84, 100, 81, 100, 100, 88, 100, 81, 90, 100, 83, 90, 81, 86, 83, 96, 82, 97, 85, 90, 81, 81, 97,
                95, 97, 84, 81, 84, 89, 86, 89, 83, 95]
ordem_classes = ['área_eletricamente_inativa',
       'Bloqueio_de_ramo_direito', 'Bloqueio_de_ramo_esquerdo',
       'Bloqueio_de_ramo_direito_e_bloqueio_divisional_anterossuperior_do_ramo_esquerdo',
       'Bloqueio_intraventricular_inespecífico',
       'Sobrecarga_ventricular_esquerda_(critérios_de_Romhilt-Estes)',
       'Sobrecarga_ventricular_esquerda_(critérios_de_voltagem)',
       'Fibrilação_atrial', 'Flutter_atrial',
       'Bloqueio_atrioventricular_de_2°_grau_Mobitz_I',
       'Bloqueio_atrioventricular_de_2°_grau_Mobitz_II',
       'Bloqueio_atrioventricular_2:1', 'Bloqueio_atrioventricular_avançado',
       'Bloqueio_atrioventricular_total',
       'Pré-excitação_ventricular_tipo_Wolff-Parkinson-White',
       'Sistema_de_estimulação_cardíaca_normofuncionante',
       'Sistema_de_estimulação_cardíaca_com_disfunção',
       'Taquicardia_atrial_multifocal', 'Taquicardia_atrial',
       'Taquicardia_supraventricular', 'Corrente_de_lesão_subendocárdica',
       'Alterações_primárias_da_repolarização_ventricular',
       'Extrassístoles_supraventriculares', 'Extrassístoles_ventriculares',
       'Bradicardia_sinusal',
       'ECG_dentro_dos_limites_da_normalidade_para_idade_e_sexo',
       'Alterações_da_repolarização_ventricular_atribuídas_à_ação_digitálica',
       'Alterações_inespecíficas_da_repolarização_ventricular',
       'Alterações_secundárias_da_repolarização_ventricular',
       'Arritmia_sinusal',
       'Ausência_de_sinal_eletrocardiográfico_que_impede_a_análise',
       'Interferência_na_linha_de_base_que_não_impede_a_análise_do_ECG',
       'Ausência_de_sinal_eletrocardiográfico_que_não_impede_a_análise',
       'Traçado_com_qualidade_técnica_insuficiente',
       'Possível_inversão_de_posicionamento_de_eletrodos',
       'Baixa_voltagem_em_derivações_precordiais',
       'Baixa_voltagem_em_derivações_periféricas',
       'Bloqueio_atrioventricular_de_1°_grau',
       'Bloqueio_de_ramo_direito_e_bloqueio_divisional_posteroinferior_do_ramo_esquerdo',
       'Bloqueio_divisional_anterossuperior_do_ramo_esquerdo',
       'Bloqueio_divisional_posteroinferior_do_ramo_esquerdo',
       'Desvio_do_eixo_do_QRS_para_direita',
       'Desvio_do_eixo_do_QRS_para_esquerda',
       'Dissociação_atrioventricular_isorrítmica',
       'Distúrbio_de_condução_do_ramo_direito',
       'Distúrbio_de_condução_do_ramo_esquerdo', 'Intervalo_PR_curto',
       'Intervalo_QT_prolongado', 'Isquemia_subendocárdica',
       'Progressão_lenta_de_R_nas_derivações_precordiais', 'Pausa_sinusal',
       'Corrente_de_lesão_subepicárdica',
       'Corrente_de_lesão_subepicárdica_-_provável_infarto_agudo_do_miocárdio_com_supradesnivelamento_de_ST',
       'Repolarização_precoce', 'Ritmo_atrial_ectópico',
       'Ritmo_atrial_multifocal', 'Ritmo_idioventricular_acelerado',
       'Ritmo_juncional', 'Síndrome_de_Brugada', 'Sobrecarga_atrial_direita',
       'Sobrecarga_atrial_esquerda', 'Sobrecarga_biatrial',
       'Sobrecarga_biventricular', 'Sobrecarga_ventricular_direita',
       'Sobrecarga_ventricular_esquerda(_critérios_de_voltagem)',
       'Taquicardia_sinusal', 'Taquicardia_ventricular_não_sustentada',
       'Taquicardia_ventricular_sustentada',
       'Suspeita_de_Síndrome_de_Brugada_repetir_V1-V2_em_derivações_superiores',
       'Taquicardia_juncional', 'Batimento_de_escape_atrial',
       'Batimento_de_escape_supraventricular', 'Batimento_de_escape_juncional',
       'Batimento_de_escape_ventricular']


# In[3]:


#ids_achados   = pd.read_csv('../../greenZoneIds.csv', index_col = 0)
#resultLabels   = pd.read_csv('../../../data/resultLabels.csv')
#baseC = pd.read_csv("../../../data/DATA_LAUDOS_TEXTO_formato1", sep = ";")


# In[4]:


# Global variables
REBUILD_DATA = False
TOP_WORDS = 7500
EPOCHS = 30 
BATCH_SIZE = 256
#Essa variável aqui é o número de posições que os vetores tem ao usar o pad_sequences. Não uso ela pra nada ainda mas talvez eu venha a usar, por isso ela está aqui.
SEQ_SIZE = 422


# In[5]:


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on a GPU :D")
else:
    device = torch.device("cpu")
    print("Running on a CPU :/")


# In[6]:


# Oi, Derick
# To comentando como diálogo porque acho mais fácil explicar o código assim. Se o comentário estiver errado é pq essa é a parte que eu entendi errado.
class Net(nn.Module):
    def __init__(self):
        
        print("Building NN...")
        embedding_dim = 128
        lstm_out_dim = 128
        num_embeddings = 7500
        num_of_classes = 74
        
        super().__init__()
        #Camada de Embedding, o padding_idx é um argumento que eu descobri que é usada para falar para a camada que os números no fim de cada vetor são apenas lixo
        self.l1 = nn.Embedding(num_embeddings, embedding_dim, padding_idx = 0)
        #Eu não entendo muito bem o que essa camada faz. Pelo que eu entendi é algo probabilístico. Mas ela n altera o shape.
        self.l2 = nn.Dropout(p=0.4)
        #A LSTM recebe os Embeddings e cospe o mesmo número de vetores que eu passei para ela. Não sei se eu deveria alterar o número de camadas da LSTM.
        #Se usar menos de 2 não dá pra colocar Dropout pq o Dropout é aplicado em todas as camadas menos na última.
        self.l3 = nn.LSTM(embedding_dim, lstm_out_dim, dropout = 0.2, num_layers = 2)
        #É o seguinte. Como as dimensões de entrada são estáticas, eu adicionei elas manualmente na camada linear para conseguir fazer o flatten.
        self.l4 = nn.Flatten()
        #Dimensao do vetor de entrada X dimensao da lstm
        self.l5 = nn.Linear(SEQ_SIZE * lstm_out_dim, num_of_classes)
        
    
    def forward(self, x):
        #Aqui eu só to passando o input pelas camadas mesmo
        x    = self.l1(x)
        x    = self.l2(x)
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
        pattern = r'[^a-zA-z0-9!.?,:\s]'
        x = normalize('NFKD', x).encode('ASCII', 'ignore').decode('ASCII')
        x = re.sub(pattern, '', x)
        return x
    else:
        return ""


# In[8]:


def tokenize(text, tokenizer, fit = False):
    # Creating vocabulary
    if fit:
        tokenizer.fit_on_texts(text)
    # Vectorizing text
    X   = tokenizer.texts_to_sequences(text)
    return X


# In[9]:


with open("../../trained_models/pytorch_tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)


# In[11]:

print("Loading data")

#Atribua aqui seu teste
y_test = np.load("../../../data/training_data/training_data_y.npy")
test_X = np.load("../../../data/training_data/training_X.npy")
# test_X_temp = [torch.Tensor(i).type(torch.LongTensor) for i in test_X_temp]


# In[23]:


test_X = np.reshape(test_X, (625940, 422))
test_X = torch.tensor(test_X)


# In[ ]:


net = Net()
net.load_state_dict(torch.load("../../trained_models/pytorch_checkpoint_2.pth",  map_location=torch.device('cpu')))
net.eval()


# In[ ]:

print("Started working!")
finalResult = torch.Tensor()
size = 64
for i in tqdm(range(0, test_X.size()[0], size)):
    with torch.no_grad():
        result      = net(test_X[i: min(i+size, test_X.size()[0])])
    finalResult = torch.cat((finalResult, result), 0)


# In[ ]:

print("Converting and saving...")
y_score = np.array(finalResult)


# In[ ]:


np.save("predictions.npy", y_score)

print("Done!")