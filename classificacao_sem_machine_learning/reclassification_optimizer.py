#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Loading modules...")
import multiprocessing
from unidecode import unidecode
from joblib import Parallel, delayed
import numpy as np
import string
import pandas as pd
import time
from unicodedata import normalize
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process
from argparse import ArgumentParser
import re
from difflib import SequenceMatcher
import Levenshtein as lev


# In[2]:


from difflib import SequenceMatcher

def scoring(a,b):
    return (max(len(a),len(b)) - lev.distance(a, b)) / max(len(a),len(b)) * 100

def partial_ratio(s1, s2):
    """"Return the ratio of the most similar substring
    as a number between 0 and 100."""

    if len(s1) <= len(s2):
        shorter = s1
        longer  = s2
    else:
        shorter = s2
        longer  = s1
    
    n = len(shorter)
    if(n == 0):
        return 0
    blocks = []
    for j in range(len(longer)):
        if (longer[j] == shorter[0]):
            blocks.append([j, min(j+n, len(longer))])
        if (longer[j] == shorter[-1]):
            blocks.append([max(j-n, 0),j])
    
    #if (longer[j] == shorter[0]) or (longer[len(shorter)+j-1] == shorter[-1])

    # each block represents a sequence of matching characters in a string
    # of the form (idx_1, idx_2, len)
    # the best partial match will block align with at least one of those blocks
    #   e.g. shorter = "abcd", longer = XXXbcdeEEE
    #   block = (1,3,3)
    #   best score === ratio("abcd", "Xbcd")
    
    ratios = [(scoring(shorter, longer[block[0]:block[1]])) for block in blocks]
    if len(ratios) > 0:
        biggest_r = max(ratios)
    else:
        return 0
#     biggest_r = 0

#     for block in blocks:
#         m2 = SequenceMatcher(None, shorter, block[0])
#         r = m2.ratio()
#         if r > .95:
#             best_long_start = block[1]
#             best_long_end = len(shorter) + block[1]
#             return [best_long_start, best_long_end, 100];
#         elif r > biggest_r:
#             best_long_start = block[1]
#             best_long_end = len(shorter) + block[1]
#             biggest_r = r

    return round(biggest_r, 3)


# In[6]:


lista_corte = [95,100,
 100, 70,
 71, 100,
 67, 88,
 100, 90,
 96, 93,
 91, 87,
 84, 71,
 81, 100,
 83, 79,
 84, 78,
 82, 47,
 100,
 80, 43,
 42, 90,
 100, 68,
 69,55,
 78, 100,
 68, 57,
 58, 70,
 83, 85,
 94, 94,
 75, 92,
 84, 56,
 56, 89,
 55, 100,
 94, 69,
 67, 67,
 100, 71,
 73, 95,
 100, 62,
 100, 96,
 100, 44,
 100, 89,
 85, 71,
 100, 65,
 69, 59,
 72]

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


def clean_text(x):
    if type(x) is str:
        pattern = r'[^a-zA-z0-9!:.,?\s]'
        x = normalize('NFKD', x).encode('ASCII', 'ignore').decode('ASCII')
        x = re.sub(pattern, '', x)
        return x.lower()
    else:
        return ""


# In[13]:


starting = 0
ending   = 2000000


# In[15]:


print("Loading data...")
db = pd.read_csv("../../data/DATA_LAUDOS_TEXTO_formato1", sep = ";")
optDict = ['ecg dentro dos limites da normalidade', 'eletrocardiograma dentro dos limites da normalidade']
print("Cutting slice. From {} to {}...".format(starting, ending))
db = db[starting:ending]
db = db.reset_index()


# In[16]:


texts = [clean_text(text) for text in db["CONTEUDO"]]
# texts = [text[text.find('conclusao'):] for text in texts]


# In[40]:


print("Started working!")
batch = 1000
print("Working in batches of", batch)
ans = np.empty(0)
for i in range(0, len(db), batch):
    print(i,"/",len(db))
    t = time.time()
    scores = [max([partial_ratio(text, clean_text(diag)) for diag in optDict])          for text in texts[i:i+batch]]
    ans = np.append(ans, scores)
    print("Took", time.time() - t, "seconds!")
    np.save("optimization_normal.npy", ans)
    
print("DONE!!!")

