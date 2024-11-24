from sentence_transformers import SentenceTransformer,InputExample,losses,evaluation,util
import pandas as pd
import numpy as np
import torch
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# print all output to a file
import sys
import os
import datetime
now = datetime.datetime.now()

data = 'R8_toy'
random_pairs = 20
positive_pairs = 5
EPS = 1e-9
model_name = 'all-MiniLM-L6-v2'
# model_name = 'all-mpnet-base-v2'
data_dir = 'data/'+data+'/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
sys.stdout = open('output_'+ data+'_baseline_'+model_name+now.strftime("%Y-%m-%d-%H-%M-%S")+'.txt', 'w')
model = SentenceTransformer(model_name)


train = pd.read_csv(data_dir+'train.csv')
train = train.sample(frac=1).reset_index(drop=True)
val = pd.read_csv(data_dir+'val.csv')
train = train.sample(frac=1).reset_index(drop=True)
# train.head()
label_encoder = preprocessing.LabelEncoder()
X_train = train['text'].values
y_train = train['label'].values
y_train = label_encoder.fit_transform(y_train)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
print(class_weights)

X_val = val['text'].values
y_val = val['label'].values
y_val = label_encoder.fit_transform(y_val)
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

train_examples = make_training_pairs(X_train,y_train)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

class Eval(evaluation.SentenceEvaluator):
    def __init__(self, name: str = "", softmax_model=None, write_csv: bool = True):
        pass

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        train_emb = model.encode(X_train, batch_size=batch_size, show_progress_bar=False,convert_to_tensor=True,device=device,normalize_embeddings=True)
        val_emb = model.encode(X_val, batch_size=batch_size, show_progress_bar=False,convert_to_tensor=True,device=device,normalize_embeddings=True)
        nval = len(val_emb)
        ypred = 
        acc = accuracy_score(____________)
        print("Epoch:",epoch, "Steps:",steps, ",Acc:",acc)
        return acc

ytrain = torch.tensor(y_train, dtype=torch.long).to(device)
yval = torch.tensor(y_val, dtype=torch.long).to(device)

model.fit(train_objectives=[(train_dataloader,losses.CosineSimilarityLoss(model))], epochs=epochs,show_progress_bar=True,evaluator=Eval(), warmup_steps=warmup_steps, optimizer_params={'lr': lr})

