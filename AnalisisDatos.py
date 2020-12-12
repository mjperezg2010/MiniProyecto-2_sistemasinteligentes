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
    string_array = string_array.replace("Dengue_Grave", 0, regex=True)
    string_array = string_array.replace("Dengue_NoGrave_NoSignos", 1, regex=True)
    string_array = string_array.replace("Dengue_NoGrave_SignosAlarma", 2, regex=True)
    string_array = string_array.replace("No_Dengue", 3, regex=True)
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



    return datos


def preprocesar2(datos):
    if datos['plaquetas'][0] == datos.loc[0][1]:
        for i in range (len(datos)):
            if datos.loc[i][1] < 193000 and datos.loc[i][-1] == 1:
                datos.iloc[[i,1]] = 0
            elif datos.loc[i][1] >= 193000 and datos.loc[i][-1] == 1:
                datos.iloc[[i,1]] = 1
            elif datos.loc[i][1] < 275000 and datos.loc[i][-1] == 3:
                datos.iloc[[i,1]] = 0
            elif datos.loc[i][1] >= 275000 and datos.loc[i][-1] == 3:
                datos.iloc[[i,1]] = 1
            elif datos.loc[i][1] < 125000 and datos.loc[i][-1] == 0:
                datos.iloc[[i,1]] = 0
            elif datos.loc[i][1] >= 125000 and datos.loc[i][-1] == 0:
                datos.iloc[[i,1]] = 1
            elif datos.loc[i][1] < 198000 and datos.loc[i][-1] == 2:
                datos.iloc[[i,1]] = 0
            elif datos.loc[i][1] >= 198000 and datos.loc[i][-1] == 2:
                datos.iloc[[i,1]] = 1
    else:
        for i in range (len(datos)):
            if datos.loc[i][3] < 193000 and datos.loc[i][-1] == 1:
                datos.iloc[[i,3]] = 0
            elif datos.loc[i][3] >= 193000 and datos.loc[i][-1] == 1:
                datos.iloc[[i,3]] = 1
            elif datos.loc[i][3] < 275000 and datos.loc[i][-1] == 3:
                datos.iloc[[i,3]] = 0
            elif datos.loc[i][3] >= 275000 and datos.loc[i][-1] == 3:
                datos.iloc[[i,3]] = 1
            elif datos.loc[i][3] < 125000 and datos.loc[i][-1] == 0:
                datos.iloc[[i,3]] = 0
            elif datos.loc[i][3] >= 125000 and datos.loc[i][-1] == 0:
                datos.iloc[[i,3]] = 1
            elif datos.loc[i][3] < 198000 and datos.loc[i][-1] == 2:
                datos.iloc[[i,3]] = 0
            elif datos.loc[i][3] >= 198000 and datos.loc[i][-1] == 2:
                datos.iloc[[i,3]] = 1

    if datos['linfocitos'][0] == datos.loc[0][2]:
        for i in range (len(datos)):
            if datos.loc[i][2] < 0.41 and datos.loc[i][-1] == 1:
                datos.iloc[[i,2]] = 0
            elif datos.loc[i][2] >= 0.41 and datos.loc[i][-1] == 1:
                datos.iloc[[i,2]] = 1
            elif datos.loc[i][2] < 0.175 and datos.loc[i][-1] == 3:
                datos.iloc[[i,2]] = 0
            elif datos.loc[i][2] >= 0.175 and datos.loc[i][-1] == 3:
                datos.iloc[[i,2]] = 1
            elif datos.loc[i][2] < 0.47 and datos.loc[i][-1] == 0:
                datos.iloc[[i,2]] = 0
            elif datos.loc[i][2] >= 0.47 and datos.loc[i][-1] == 0:
                datos.iloc[[i,2]] = 1
            elif datos.loc[i][2] < 0.405 and datos.loc[i][-1] == 2:
                datos.iloc[[i,2]] = 0
            elif datos.loc[i][2] >= 0.405 and datos.loc[i][-1] == 2:
                datos.iloc[[i,2]] = 1
    else:
        for i in range(len(datos)):
            if datos.loc[i][4] < 0.41 and datos.loc[i][-1] == 1:
                datos.iloc[[i, 4]] = 0
            elif datos.loc[i][4] >= 0.41 and datos.loc[i][-1] == 1:
                datos.iloc[[i, 4]] = 1
            elif datos.loc[i][4] < 0.175 and datos.loc[i][-1] == 3:
                datos.iloc[[i, 4]] = 0
            elif datos.loc[i][4] >= 0.175 and datos.loc[i][-1] == 3:
                datos.iloc[[i, 4]] = 1
            elif datos.loc[i][4] < 0.47 and datos.loc[i][-1] == 0:
                datos.iloc[[i, 4]] = 0
            elif datos.loc[i][4] >= 0.47 and datos.loc[i][-1] == 0:
                datos.iloc[[i, 4]] = 1
            elif datos.loc[i][4] < 0.405 and datos.loc[i][-1] == 2:
                datos.iloc[[i,4]] = 0
            elif datos.loc[i][4] >= 0.405 and datos.loc[i][-1] == 2:
                datos.iloc[[i,4]] = 1

    if datos['hematocritos'][0] == datos.loc[0][3]:
        for i in range (len(datos)):
            if datos.loc[i][3] < 0.44 and datos.loc[i][-1] == 1:
                datos.iloc[[i,3]] = 0
            elif datos.loc[i][3] >= 0.44 and datos.loc[i][-1] == 1:
                datos.iloc[[i,3]] = 1
            elif datos.loc[i][3] < 0.422 and datos.loc[i][-1] == 3:
                datos.iloc[[i,3]] = 0
            elif datos.loc[i][3] >= 0.422 and datos.loc[i][-1] == 3:
                datos.iloc[[i,3]] = 1
            elif datos.loc[i][3] < 0.452 and datos.loc[i][-1] == 0:
                datos.iloc[[i,3]] = 0
            elif datos.loc[i][3] >= 0.452 and datos.loc[i][-1] == 0:
                datos.iloc[[i,3]] = 1
            elif datos.loc[i][3] < 0.43 and datos.loc[i][-1] == 2:
                datos.iloc[[i,3]] = 0
            elif datos.loc[i][3] >= 0.43 and datos.loc[i][-1] == 2:
                datos.iloc[[i,3]] = 1
    else:
        for i in range(len(datos)):
            if datos.loc[i][5] < 0.41 and datos.loc[i][-1] == 1:
                datos.iloc[[i, 5]] = 0
            elif datos.loc[i][5] >= 0.41 and datos.loc[i][-1] == 1:
                datos.iloc[[i, 5]] = 1
            elif datos.loc[i][5] < 0.175 and datos.loc[i][-1] == 3:
                datos.iloc[[i, 5]] = 0
            elif datos.loc[i][5] >= 0.175 and datos.loc[i][-1] == 3:
                datos.iloc[[i, 5]] = 1
            elif datos.loc[i][5] < 0.47 and datos.loc[i][-1] == 0:
                datos.iloc[[i, 5]] = 0
            elif datos.loc[i][5] >= 0.47 and datos.loc[i][-1] == 0:
                datos.iloc[[i, 5]] = 1
            elif datos.loc[i][5] < 0.405 and datos.loc[i][-1] == 2:
                datos.iloc[[i,5]] = 0
            elif datos.loc[i][5] >= 0.405 and datos.loc[i][-1] == 2:
                datos.iloc[[i,5]] = 1

    if datos['leucocitos'][0] == datos.loc[0][3]:
        for i in range (len(datos)):
            if datos.loc[i][4] < 6100 and datos.loc[i][-1] == 1:
                datos.iloc[[i,4]] = 0
            elif datos.loc[i][4] >= 6100 and datos.loc[i][-1] == 1:
                datos.iloc[[i,4]] = 1
            elif datos.loc[i][4] < 8250 and datos.loc[i][-1] == 3:
                datos.iloc[[i,4]] = 0
            elif datos.loc[i][4] >= 8250 and datos.loc[i][-1] == 3:
                datos.iloc[[i,4]] = 1
            elif datos.loc[i][4] < 5750 and datos.loc[i][-1] == 0:
                datos.iloc[[i,4]] = 0
            elif datos.loc[i][4] >= 5750 and datos.loc[i][-1] == 0:
                datos.iloc[[i,4]] = 1
            elif datos.loc[i][4] < 6250 and datos.loc[i][-1] == 2:
                datos.iloc[[i,4]] = 0
            elif datos.loc[i][4] >= 6250 and datos.loc[i][-1] == 2:
                datos.iloc[[i,4]] = 1
    else:
        for i in range(len(datos)):
            if datos.loc[i][6] < 6100 and datos.loc[i][-1] == 1:
                datos.iloc[[i, 6]] = 0
            elif datos.loc[i][6] >= 6100 and datos.loc[i][-1] == 1:
                datos.iloc[[i, 6]] = 1
            elif datos.loc[i][6] < 8250 and datos.loc[i][-1] == 3:
                datos.iloc[[i, 6]] = 0
            elif datos.loc[i][6] >= 8250 and datos.loc[i][-1] == 3:
                datos.iloc[[i, 6]] = 1
            elif datos.loc[i][6] < 5750 and datos.loc[i][-1] == 0:
                datos.iloc[[i, 6]] = 0
            elif datos.loc[i][6] >= 5750 and datos.loc[i][-1] == 0:
                datos.iloc[[i, 6]] = 1
            elif datos.loc[i][6] < 6250 and datos.loc[i][-1] == 2:
                datos.iloc[[i, 6]] = 0
            elif datos.loc[i][6] >= 6250 and datos.loc[i][-1] == 2:
                datos.iloc[[i, 6]] = 1

    print(datos['leucocitos'])



def main():
    #datos=pd.read_csv("laboratorio_train_synth_dengue.csv")
    #print(datos.loc[0][-1])
    datos=pd.read_csv("completo_train_synth_dengue.csv")
    datos=preprocesar(datos)
    preprocesar2(datos)
    #print(datos.head())
    #normalizacion(datos)


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


