import Levenshtein as lev
from unicodedata import normalize
import re
import torch.nn as nn
import torch.nn.functional as F

def scoring(a,b):
    if len(b) < max(len(a) - 2, 3): return 0
    dim = 0
    if b[-1] == '?' or b[-2] == '?': dim = 40
    return ((max(len(a),len(b)) - lev.distance(a, b)) / max(len(a),len(b)) * 100) - dim

def make_score(s1, s2):
    """"Return the ratio of the most similar substring
    as a number between 0 and 100."""

    if len(s1) <= len(s2):
        shorter = s1
        longer  = s2
    else:
        shorter = s2
        longer  = s1
    
    n = len(shorter)
    
    if(n < 2):
        return 0
    
    if(shorter[-1] == '!'):
        longer = " " + longer + " "
        shorter = " " + shorter[:-1] + " "
        
        if(longer.find(shorter) != -1) or (longer.find("(" + shorter[1:-1] + ")") != -1):
            return 100
        else:
            return 0
    else:
        acronym = False
        
    if(shorter[-1] == "#"):
        shorter = shorter[:-1]
        precise = True
    else:
        precise = False
    
    blocks = []
    for j in range(len(longer)):
        if (longer[j] == shorter[0]):
            blocks.append([max(0, j), min(j+n+2, len(longer))])
        if (longer[j] == shorter[-1]):
            blocks.append([max(j-n, 0),min(j+2, len(longer))])
    
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
    
    
    if precise and biggest_r < 75: biggest_r = 0    
    return round(biggest_r, 3)

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

def clean_text(x):
    if type(x) is str:
        pattern = r'[^a-zA-z0-9!.,?\s]'
        x = normalize('NFKD', x).encode('ASCII', 'ignore').decode('ASCII')
        x = re.sub(pattern, '', x)
        return x.lower()
    else:
        return ""


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Net(nn.Module):
    
    def __init__(self, seq_size, top_words, bidirectional = True):
        
        print("Building NN...")
        embedding_dim = 128
        lstm_out_dim = 128
        num_embeddings = top_words
        num_of_classes = 35
        
        super().__init__()
        #Camada de Embedding, o padding_idx é um argumento que eu descobri que é usada para falar para a camada que os números no fim de cada vetor são apenas lixo
        self.l1 = nn.Embedding(num_embeddings, embedding_dim, padding_idx = 0)
        #Eu não entendo muito bem o que essa camada faz. Pelo que eu entendi é algo probabilístico. Mas ela n altera o shape.
#         self.l2 = nn.Dropout(p=0.4)
        #A LSTM recebe os Embeddings e cospe o mesmo número de vetores que eu passei para ela. Não sei se eu deveria alterar o número de camadas da LSTM.
        #Se usar menos de 2 não dá pra colocar Dropout pq o Dropout é aplicado em todas as camadas menos na última.
        self.l3 = nn.LSTM(embedding_dim, lstm_out_dim, dropout = 0.2, num_layers = 2, bidirectional = bidirectional)
        #É o seguinte. Como as dimensões de entrada são estáticas, eu adicionei elas manualmente na camada linear para conseguir fazer o flatten.
        self.l4 = nn.Flatten()
        #Dimensao do vetor de entrada X dimensao da lstm
        self.l5 = nn.Linear(seq_size * lstm_out_dim * (2 if bidirectional else 1), num_of_classes)
        
    
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
