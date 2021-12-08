#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import pickle
from unicodedata import normalize
import os
import sys
import re
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import transformers
# import textclassification as tc


# In[3]:


def clean_text(x):
    if type(x) is str:
        pattern = r'[^a-zA-z0-9!.,?\s]'
        x = normalize('NFKD', x).encode('ASCII', 'ignore').decode('ASCII')
        x = re.sub(pattern, '', x)
        return x.lower()
    else:
        return ""


# In[4]:


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


# In[5]:


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)



# In[6]:


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# In[7]:

print("Loading X...")
vocab_size = 10000  # Only consider the top 20k words
maxlen = 411  # Only consider the first 200 words of each movie review
x_train = np.load("data/x_train.npy", allow_pickle = True)
print(len(x_train), "Training sequences")
# print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)


# In[8]:

print("Loading Y...")
y_train = np.load("data/score_bin.npy", allow_pickle = True)

gZoneIdx = np.load("data/gZoneIdx.npy", allow_pickle = True)
print("Done! GreenZone has", len(gZoneIdx), "registers!")

removeIdx = []
# Finding removeIdx in O(n) using two pointers :D
l = 0
n = 0
while(l < len(gZoneIdx)):
    if(gZoneIdx[l] != n):
        removeIdx.append(n)
    else: l += 1
    n+=1
print("Removing useless entries...")
y_train = np.delete(y_train, removeIdx, axis = 0)


# In[28]:


embed_dim = 32  # Embedding size for each token
num_heads = 8  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
out_dim = 35

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block1 = TransformerBlock(embed_dim, num_heads, ff_dim)
transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim)
transformer_block3 = TransformerBlock(embed_dim, num_heads, ff_dim)
transformer_block4 = TransformerBlock(embed_dim, num_heads, ff_dim)
transformer_block5 = TransformerBlock(embed_dim, num_heads, ff_dim)
transformer_block6 = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block1(x)
x = transformer_block2(x)
x = transformer_block3(x)
x = transformer_block4(x)
x = transformer_block5(x)
x = transformer_block6(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
x = layers.Flatten()(x)
outputs = layers.Dense(35, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)


# In[11]:


test_df = pd.read_csv("../light_data/gold_standard.csv")
text = test_df["texto"]


# In[27]:


print("Finding test on train...")
train_df = pd.read_csv("../../data/DATA_LAUDOS_TEXTO_formato1", sep = ";")
train_df = train_df[:2000000]
train_df = train_df[train_df.index.isin(gZoneIdx)].reset_index()
rmv_test = train_df[train_df["ID_EXAME"].isin(test_df["id_exame"])].index
rmv_test = np.array(rmv_test)
print("Removing test from train...")
y_train = np.delete(y_train, rmv_test, axis = 0)
x_train = np.delete(x_train, rmv_test, axis = 0)
print("Have it worked?", "Yes" if len(y_train) == len(x_train) else "No")
print("Length ->",len(y_train))


# In[30]:


#Okay, training
print("Okay let's train:\n\n")
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=128, epochs=10, validation_split=0.1,
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min'),
             tf.keras.callbacks.ModelCheckpoint(filepath='best_model', monitor='val_loss', mode='min', save_best_only=True)]
)


# In[29]:

print("Saving...")
model.save("last_model")
print("Finish! :D")