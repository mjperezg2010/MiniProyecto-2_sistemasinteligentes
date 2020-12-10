import numpy as np
import csv
import random
from sklearn.metrics import f1_score
import pandas as pd

def load_file(file_name):
    return pd.read_csv(file_name)
    #return list(csv.reader(open(file_name), delimiter=','))

def process_data(file_name):    
    string_array = load_file(file_name)
    string_array=string_array.replace(np.nan,'NA',regex=True)
    string_array=string_array.replace("NO", "No", regex=True)
    string_array=string_array.replace("F",0,regex=True)
    string_array=string_array.replace("M", 1, regex=True)
    string_array=string_array.replace("Si", 0, regex=True)
    string_array=string_array.replace("No", 1, regex=True)
    string_array=string_array.replace("Persistente", 2, regex=True)
    string_array=string_array.replace("NA", -1, regex=True)
    string_array=string_array.replace("Positiva", 0, regex=True)
    string_array=string_array.replace("Negativa", 1, regex=True)

    X = string_array.loc[:, string_array.columns != 'clase']
    Y = string_array['clase']
    return X,Y


def split_data(n_groups,X,Y):
    groups_X = []
    groups_Y = []
    temp = [x for x in range(len(X))]
    for _ in range(20):
        random.shuffle(temp)
    
    for i in range(n_groups):
        actual_X = []
        actual_Y = []
        for j in range(int(len(X)/n_groups)):
            actual_X.append(X.values[temp[i*int(len(X)/n_groups)+j]])
            actual_Y.append(Y.values[temp[i*int(len(X)/n_groups)+j]])
        groups_X.append(actual_X)
        groups_Y.append(actual_Y)
    return groups_X,groups_Y

def print_data(a, b, c, d, e, i, F1):
    print("dataset:"+str(a)+" n_trees: "+str(b)+" criterio: "+str(c)+
        " max_depth: "+str(d)+" n_features: "+str(e)+" model_num: "+str(i)+" F1: "+str(F1))

def F1(model,x,y):
    predicted = model.predict(x)
    results = f1_score(predicted, y, average=None)
    acum = 0
    total = len(results)
    for i in results:
        acum = acum + i

    return acum / total