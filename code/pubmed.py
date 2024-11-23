import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer,InputExample,losses,evaluation,util


EPS = 1e-9
model_name = '____________________'


data = 'pubmed'
data_dir = ### add path of dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print('Using device:', device)


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

model = SentenceTransformer(model_name, device=device)

df_edges = pd.read_csv(data_dir+'pubmed_edges.tsv', sep = '\t', header = None)

random_pairs = 20
positive_pairs = 5

df_text = pd.read_csv(data_dir+'pubmed_full.tsv',sep = '\t')

train = df_text[df_text['split']=='train']
val = df_text[df_text['split']=='val']

X_train = (train['TI'].fillna('') + ' ' + train['AB'].fillna('')).values
y_train = train['label'].values
id_train = train['PMID'].values

X_val = (val['TI'].fillna('') + ' ' + val['AB'].fillna('')).values
y_val = val['label'].values
id_val = val['PMID'].values

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
num_labels = len(np.unique(y_train))
print(num_labels)

def make_training_pairs(X,y):
    train_examples = []    
    # select 20 random sentences and make pairs accordingly(both labels 1,0) 
    # select 5 sentences from same class as s, 
    for i in range(len(X)):
        random_indices = np.random.choice(len(X), random_pairs, replace=False)
        for j in random_indices:
            train_examples.append(InputExample(texts=[X[i], X[j]], label=float(y[i]==y[j])))
        same_class_indices = np.where(y==y[i])[0]
        random_indices = np.random.choice(same_class_indices, positive_pairs, replace=True)
        for j in random_indices:
            train_examples.append(InputExample(texts=[X[i], X[j]], label=1.0))
    return train_examples

print("No. of training pairs:", len(make_training_pairs(X_train,y_train)))