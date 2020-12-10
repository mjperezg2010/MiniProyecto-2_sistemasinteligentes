import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import KFold


def is_number(a):
    a = 12.02
    if (type(a) == float):
        print('This is a float')
    else:
        print('Not a float')

def matriz(datos,column):
    temp = pd.crosstab(datos[column], datos['clase'])
    print(column)
    print(temp.to_string())


def BoxPlot(datos,column):
    plt.figure(figsize=(40,15))
    seaborn.set(font_scale=1.5)
    seaborn.boxplot(x=datos[column],y=datos['clase'],data=datos)
    plt.show()

def preprocesar(datos):
    datos=datos.replace(np.nan,'NA',regex=True)
    datos=datos.replace("NO","No",regex=True)
    return datos

def split(datos):
    string_array=datos
    string_array = string_array.replace(np.nan, 'NA', regex=True)
    string_array = string_array.replace("NO", "No", regex=True)
    string_array = string_array.replace("F", 0, regex=True)
    string_array = string_array.replace("M", 1, regex=True)
    string_array = string_array.replace("Si", 0, regex=True)
    string_array = string_array.replace("No", 1, regex=True)
    string_array = string_array.replace("Persistente", 2, regex=True)
    string_array = string_array.replace("NA", -1, regex=True)
    string_array = string_array.replace("Positiva", 0, regex=True)
    string_array = string_array.replace("Negativa", 1, regex=True)
    datos=string_array


    X = datos.loc[:, datos.columns != 'clase']
    y = datos['clase']

    groups_X = []
    groups_Y = []


    kf=KFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        groups_X.append(X_train)
        groups_Y.append(y_train)

    return groups_X,groups_Y


def split_data(datos):
    string_array = datos
    string_array = string_array.replace(np.nan, 'NA', regex=True)
    string_array = string_array.replace("NO", "No", regex=True)
    string_array = string_array.replace("F", 0, regex=True)
    string_array = string_array.replace("M", 1, regex=True)
    string_array = string_array.replace("Si", 0, regex=True)
    string_array = string_array.replace("No", 1, regex=True)
    string_array = string_array.replace("Persistente", 2, regex=True)
    string_array = string_array.replace("NA", -1, regex=True)
    string_array = string_array.replace("Positiva", 0, regex=True)
    string_array = string_array.replace("Negativa", 1, regex=True)
    datos = string_array

    X = datos.loc[:, datos.columns != 'clase']
    y = datos['clase']


    groups_X = []
    groups_Y = []
    temp = [x for x in range(len(X))]
    for _ in range(20):
        random.shuffle(temp)

    for i in range(5):
        actual_X = []
        actual_Y = []
        for j in range(int(len(X) / 5)):
            actual_X.append(X.values[temp[i * int(len(X.values) / 5) + j]])
            actual_Y.append(y.values[temp[i * int(len(X.values) / 5) + j]])
        groups_X.append(actual_X)
        groups_Y.append(actual_Y)
    return groups_X, groups_Y


def main():
    datos=pd.read_csv("clinica_train_synth_dengue.csv")
    datos=preprocesar(datos)
    split(datos)
    #split_data(datos)

    """
    for column in datos.columns:
        if column == 'clase':
            break
        if (str)(datos[column][0]).find(".")!=-1:
            BoxPlot(datos,column)
        else:
            matriz(datos,column)
    """




if __name__ == '__main__':
    main()


