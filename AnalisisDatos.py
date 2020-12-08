import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    sns.set(font_scale=1.5)
    sns.boxplot(x=datos[column],y=datos['clase'],data=datos)
    plt.show()

def preprocesar(datos):
    datos=datos.replace(np.nan,'NA',regex=True)
    datos=datos.replace("NO","No",regex=True)
    return datos


def main():
    datos=pd.read_csv("completo_train_synth_dengue.csv")
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


