import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


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

    #Borrar ahi
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

    string_array = string_array.replace("Dengue_Grave", "0", regex=True)
    string_array = string_array.replace("Dengue_NoGrave_NoSignos", "1", regex=True)
    string_array = string_array.replace("Dengue_NoGrave_SignosAlarma", "2", regex=True)
    string_array = string_array.replace("No_Dengue", "3", regex=True)
    datos=string_array



    return datos


def normalizacion(datos):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(datos)
    datos = pd.DataFrame(scaled)
    print(datos.head())


def main():
    datos=pd.read_csv("clinica_train_synth_dengue.csv")
    datos=preprocesar(datos)
    print(datos.head())
    normalizacion(datos)


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


