
# coding: utf-8

# # Deep Mailing - XGBoost Model - Reduzindo Dimensões

# O objetivo desse notebook é demonstrar a utilizacao do XGBoost para a criacao de arvores de decisão para a predição de CUPS em mailings.
# 
# Em primeiro lugar, definimos os imports que iremos usar...

# In[1]:


import xgboost
import numpy as np
import os
import sys
import logging
import gc
import pickle as pickle
import pandas as pd
import dateutil.parser as parser
import os.path
import math
from sklearn.metrics import accuracy_score,precision_score,recall_score
from datetime import datetime
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams


# Abaixo definimos os diretorios e nomes dos arquivos intermediarios.

# In[2]:


log_location = "../logs/"
arquivo_df_pickled_norm = "../intermediate/df.norm.corr.pickle"
arquivo_df_pickled_norm_train = "../intermediate/df.norm.train.pickle"
arquivo_df_pickled_norm_test = "../intermediate/df.norm.test.pickle"
arquivo_df_pickled_norm_train_x = "../intermediate/df.norm.train.x.pickle.npy"
arquivo_df_pickled_norm_train_y = "../intermediate/df.norm.train.y.pickle.npy"
arquivo_df_pickled_norm_test_x = "../intermediate/df.norm.test.x.pickle.npy"
arquivo_df_pickled_norm_test_y = "../intermediate/df.norm.test.y.pickle.npy"


# Definimos o versampling que é o minimo de rows com a variavel alvo setado no dataset de treinamento

# In[3]:


Oversampling = False
RateOversampling = 0.3


# Redefinimos o logger que iremos usar

# In[4]:


logger = logging.getLogger()
logging.basicConfig(format="%(asctime)-15s %(message)s",
                    level=logging.DEBUG,
                    filename=os.path.join(log_location,'xgboost.log.' + datetime.now().strftime("%Y%m%d%H%M%S.%f") + '.log'))


# Criamos uma função para imprimir tanto no log quanto no notebook...

# In[5]:


def print_log(msg):
    logging.debug(msg)
    print(msg)    


# Carregamos para a memoria o arquivo normalizado e pickled que foi gerado no notebook de "Preparação de Dados".

# In[6]:


print_log("Carregando Pickling normalizado:{}".format(arquivo_df_pickled_norm))    
chamadas = pd.read_pickle(arquivo_df_pickled_norm)


# Verificamos as dimensões do dataframe carregado.... E imprimimos uma amostra do dado que precisamos com apenas as colunas relevantes... Como podemos perceber o nosso modelo considera apenas colunas com valores booleanos (0 ou 1)

# In[7]:


chamadas.head(10)


# In[8]:


print_log(chamadas.shape)
chamadas.loc[:, :'NORM_LIGACOES'].head(10)


# Para tratarmos as colunas de forma correta, precisamos eliminar os caracteres especiais das colunas, principalmente o espaco e o sinal de - para isso iremos rodar o codigo abaixo para tratar.

# In[9]:


cols = chamadas.columns
cols = cols.map(lambda x: x.replace(' ', '_').replace('-','menos'))
chamadas.columns = cols


# Podemos agora analisar o tamanho final do nosso dataframe:

# In[10]:


total_chamadas = len(chamadas.index)
print_log("Total chamadas:{}".format(total_chamadas))


# Perfeito, então vamos embaralhar os dados e gerar os nossos dados de treinamento e teste. Vamos considerar 70% para treinamento e 30 % para teste. No final, apagamos o dataframe lido para economizar memoria

# In[11]:


print_log("Criando Pickling de train e teste...")
chamadas = chamadas.sample(int(len(chamadas.index)))
chamadas_train = chamadas.tail(int(len(chamadas.index) * 0.7))
chamadas_test = chamadas.head(int(len(chamadas.index) * 0.3))
del chamadas


# Criamos uma função para gerar um arquivo de referencia de colunas a serem usadas q que vai ser importante na hora de gerar a arvore de decisao...

# In[12]:


def create_column_reference(header_chamadas_x,arquivo_df_pickled_norm_train_x):
    print_log("Criando Arquivo de referencia de colunas...")
    with open(arquivo_df_pickled_norm_train_x+".txt","w") as f:
        counter = 0
        lista_header = list(header_chamadas_x.columns.values)
        for header in lista_header:
            f.write("{}-{}\n".format(counter,header))
            counter=counter+1


# Criamos agora os nossos dataframes de X que são as features e as Y que são os alvos de predição. Também removemos qualquer linha em tentativas seja igual a zero, além de criar um arquivo de referencia com as colunas X que serão usadas no modelo. Esses dataframes serão convertidos para matrizes no formato numpy

# In[13]:


print_log("Separando colunas em X e Y...")        
create_column_reference(chamadas_train.loc[:, :'NORM_LIGACOES'].head(1), arquivo_df_pickled_norm_train_x)
chamadas_train_x = chamadas_train.loc[:, :'NORM_LIGACOES'].as_matrix()
chamadas_train_y = chamadas_train.NORM_CUP.as_matrix()
chamadas_test_x = chamadas_test.loc[:, :'NORM_LIGACOES'].as_matrix()
chamadas_test_y = chamadas_test.NORM_CUP.as_matrix()


# Após a criação das matrizes numpy, gravamos elas em arquivos.

# In[14]:


print_log("Criando arquivos finais em formato NUMPY para consumo pelo algoritmo...")        
np.save(arquivo_df_pickled_norm_train_x,chamadas_train_x)
np.save(arquivo_df_pickled_norm_train_y,chamadas_train_y)
np.save(arquivo_df_pickled_norm_test_x,chamadas_test_x)
np.save(arquivo_df_pickled_norm_test_y,chamadas_test_y)


# Apagamos todos os dados intermediários e rodamos o garbage collector para economizar memória.

# In[15]:


print_log("Removendo objetos desnecessarios")        
del chamadas_train_x
del chamadas_train_y
del chamadas_train
del chamadas_test
del chamadas_test_x
del chamadas_test_y
gc.collect()


# Carregamos os objetos numpy em memoria

# In[16]:


print_log("Carregando objetos numpy")        
train_x = np.load(arquivo_df_pickled_norm_train_x)
train_y = np.load(arquivo_df_pickled_norm_train_y)
test_x = np.load(arquivo_df_pickled_norm_test_x)
test_y = np.load(arquivo_df_pickled_norm_test_y)


# Contamos quantos CUPS existem em treinamento e teste...

# In[17]:


msg1 = "Train - CUPS Detectados {} num universo de {}".format(len([y for y in train_y if y >0]),len(train_y))
msg2 = "Test - CUPS Detectados {} num universo de {}".format(len([y for y in test_y if y >0]),len(test_y))
print_log(msg1)
print_log(msg2)


# Configuramos os parametros para o XGBoost, especificando que queremos que ele seja o mais exato possivel, que a medida de erro é erro simples e que queremos apenas uma classificacao binaria com um maximo de 1000 interações

# In[18]:


param = {}
param['eta'] = 0.3
param['objective'] = 'binary:logistic'
param['eval_metric'] = 'auc'
param['tree_method'] = 'exact'
param['silent'] = 0
param['max_depth'] = 10
num_round = 1000


# Após a definicão dos paramêtros de teste, criamos as matrizes no formato do XGBoost e treinamos o modelo.

# In[19]:


gc.collect()
print_log("Starting model for params:{}".format(param))
dtrain = xgb.DMatrix(train_x, train_y)
dtest = xgb.DMatrix(test_x, test_y)


# In[ ]:


train_labels = dtrain.get_label()
ratio = float(np.sum(train_labels == 0)) / np.sum(train_labels == 1) 
param['scale_pos_weight'] = ratio
print_log("ratio:{}".format(ratio))


# In[ ]:


gpu_res = {}
booster = xgb.train(param, dtrain, num_round, evals=[], evals_result=gpu_res)


# Após o modelo ser treinado, podemos plotar ele... Para verificar que coluna é cada feature no modelo, por favor ver a lista em anexo no final desse notebook.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 80,50
plot_tree(booster, rankdir='LR')
plt.show()


# Agora, vamos tentar predizer os dados com o nosso modelo treinado...

# In[ ]:


test_y_pred = booster.predict(dtest)
test_predictions = np.array([value for value in test_y_pred])


# E Finalmente medir a precisão da nossa predição... Tanto no total quanto em CUPs detectados.

# In[ ]:


accuracy = accuracy_score(test_y, test_predictions.round())
precision = precision_score(test_y, test_predictions.round())
recall = recall_score(test_y, test_predictions.round())

print_log("CUPS Previstos:{}".format(len([x for x in test_predictions if x > 0.5])))
print_log("CUPS na Base Teste:{}".format(len([x for x in test_y if x > 0.5])))
print_log("Accuracy Total:{}".format(accuracy))
print_log("Accuracy em CUPs:{}".format(len([x for x in test_predictions if x > 0.5]) / len([x for x in test_y if x > 0.5])))
print_log("Precision:{}".format(precision))
print_log("Recall:{}".format(recall))


# Após, vamos salvar o modelo gerado em um arquivo para reuso...

# In[ ]:


save_file = "../output/{}.model".format(datetime.now().strftime("%Y%m%d.%H%M%S"))
with open(save_file, 'wb') as fp:
    pickle.dump(booster, fp)    
print_log("Model saved as {}".format(save_file))


# In[ ]:


get_ipython().run_cell_magic('bash', '', '\ncat ../intermediate/df.norm.train.x.pickle.npy.txt  ')

