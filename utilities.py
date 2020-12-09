import numpy as np
import csv
import random
from sklearn.metrics import f1_score

def load_file(file_name):
    return list(csv.reader(open(file_name), delimiter=','))

def process_data(file_name):    
    string_array = np.array(load_file(file_name)[1:][:])
    
    Y = [ row [string_array.shape[1]-1 ] for row in string_array]
    X = np.delete(string_array, string_array.shape[1]-1, 1)

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
            actual_X.append(X[temp[i*int(len(X)/n_groups)+j]])
            actual_Y.append(Y[temp[i*int(len(X)/n_groups)+j]])
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