

import numpy as np
import csv
import random

from sklearn.ensemble import RandomForestClassifier as nicolle

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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

def main():
    X,Y = process_data("clinica_train_synth_dengue.csv")

    trees = 1
    criterion = ['gini','entropy']
    max_depth = 1
    max_features = ['auto', 'log2']
    
    model = nicolle(
        n_estimators = trees,
        criterion = criterion,
        max_depth = max_depth,
        max_features = max_features
    )

    groups = 5

    groups_X,groups_Y = split_data(groups,X,Y)

    for i in range(groups):
        validation_X = groups_X[i]
        validation_Y = groups_Y[i]
        for j in range(groups):
            if j != i:
                training_X += groups_X[i]
                training_Y += groups_Y[i]
        model.fit(X,Y)
