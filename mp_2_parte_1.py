import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys

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


def main():

    datos=pd.read_csv(sys.argv[1])
    datos=preprocesar(datos)

    for column in datos.columns:
        if column == 'clase':
            break
        if (str)(datos[column][0]).find(".")!=-1:
            BoxPlot(datos,column)
        else:
            matriz(datos,column)

if __name__ == '__main__':
    main()


