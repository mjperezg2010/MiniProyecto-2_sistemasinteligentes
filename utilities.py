import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import statistics as stats
import seaborn
import matplotlib.pyplot as plt


# Funciones

# Funcion para cargar el archivo
def load_file(file_name):
    return pd.read_csv(file_name)


# Funcion de procesamiento de los datos
def process_data(file_name):
    string_array = load_file(file_name)
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

    X = string_array.loc[:, string_array.columns != 'clase']
    Y = string_array['clase']

    return X, Y


def preprocesar2(file):
    datos = load_file(file)

    datos = datos.replace("Dengue_Grave", 0, regex=True)
    datos = datos.replace("Dengue_NoGrave_NoSignos", 1, regex=True)
    datos = datos.replace("Dengue_NoGrave_SignosAlarma", 2, regex=True)
    datos = datos.replace("No_Dengue", 3, regex=True)
    datos = datos.replace(np.nan, 'NA', regex=True)
    datos = datos.replace("NO", "No", regex=True)
    datos = datos.replace("F", 0, regex=True)
    datos = datos.replace("M", 1, regex=True)
    datos = datos.replace("Si", 0, regex=True)
    datos = datos.replace("No", 1, regex=True)
    datos = datos.replace("Persistente", 2, regex=True)
    datos = datos.replace("NA", -1, regex=True)
    datos = datos.replace("Positiva", 0, regex=True)
    datos = datos.replace("Negativa", 1, regex=True)

    if len(datos.loc[0]) != 18:


        lista=['plaquetas','linfocitos','hematocritos','leucocitos']
        cont=1
        cont2=3
        for l in lista:
            df = datos[datos['clase'] == 0][l]
            med0 = stats.median(df)
            df = datos[datos['clase'] == 1][l]
            med1 = stats.median(df)
            df = datos[datos['clase'] == 2][l]
            med2 = stats.median(df)
            df = datos[datos['clase'] == 3][l]
            med3 = stats.median(df)

            if datos[l][0] == datos.loc[0][cont]:
                for i in range(len(datos)):
                    if datos.loc[i][cont] < med1 and datos.loc[i][-1] == 1:
                        datos.iloc[i, cont] = 0
                    elif datos.loc[i][cont] >= med1 and datos.loc[i][-1] == 1:
                        datos.iloc[i, cont] = 1
                    elif datos.loc[i][cont] < med3 and datos.loc[i][-1] == 3:
                        datos.iloc[i, cont] = 0
                    elif datos.loc[i][cont] >= med3 and datos.loc[i][-1] == 3:
                        datos.iloc[i, cont] = 1
                    elif datos.loc[i][cont] < med0 and datos.loc[i][-1] == 0:
                        datos.iloc[i, cont] = 0
                    elif datos.loc[i][cont] >= med0 and datos.loc[i][-1] == 0:
                        datos.iloc[i, cont] = 1
                    elif datos.loc[i][cont] < med2 and datos.loc[i][-1] == 2:
                        datos.iloc[i, cont] = 0
                    elif datos.loc[i][cont] >= med2 and datos.loc[i][-1] == 2:
                        datos.iloc[i, cont] = 1
            else:
                for i in range(len(datos)):
                    if datos.loc[i][cont2] < med1 and datos.loc[i][-1] == 1:
                        datos.iloc[i, cont2] = 0
                    elif datos.loc[i][cont2] >= med1 and datos.loc[i][-1] == 1:
                        datos.iloc[i, cont2] = 1
                    elif datos.loc[i][cont2] < med3 and datos.loc[i][-1] == 3:
                        datos.iloc[i, cont2] = 0
                    elif datos.loc[i][cont2] >= med3 and datos.loc[i][-1] == 3:
                        datos.iloc[i, cont2] = 1
                    elif datos.loc[i][cont2] < med0 and datos.loc[i][-1] == 0:
                        datos.iloc[i, cont2] = 0
                    elif datos.loc[i][cont2] >= med0 and datos.loc[i][-1] == 0:
                        datos.iloc[i, cont2] = 1
                    elif datos.loc[i][cont2] < med2 and datos.loc[i][-1] == 2:
                        datos.iloc[i, cont2] = 0
                    elif datos.loc[i][cont2] >= med2 and datos.loc[i][-1] == 2:
                        datos.iloc[i, cont2] = 1

            cont = cont +1
            cont2 = cont2 + 1


        """
        df = datos[datos['clase'] == 0]['plaquetas']
        med0 = stats.median(df)
        df = datos[datos['clase'] == 1]['plaquetas']
        med1 = stats.median(df)
        df = datos[datos['clase'] == 2]['plaquetas']
        med2 = stats.median(df)
        df = datos[datos['clase'] == 3]['plaquetas']
        med3 = stats.median(df)

        if datos['plaquetas'][0] == datos.loc[0][1]:
            print("entro-plaquetas")
            for i in range(len(datos)):
                if datos.loc[i][1] < med1 and datos.loc[i][5] == 1:
                    datos.iloc[i, 1] = 0
                elif datos.loc[i][1] >= med1 and datos.loc[i][5] == 1:
                    datos.iloc[i, 1] = 1
                elif datos.loc[i][1] < med3 and datos.loc[i][5] == 3:
                    datos.iloc[i, 1] = 0
                elif datos.loc[i][1] >= med3 and datos.loc[i][5] == 3:
                    datos.iloc[i, 1] = 1
                elif datos.loc[i][1] < med0 and datos.loc[i][5] == 0:
                    datos.iloc[i, 1] = 0
                elif datos.loc[i][1] >= med0 and datos.loc[i][5] == 0:
                    datos.iloc[i, 1] = 1
                elif datos.loc[i][1] < med2 and datos.loc[i][5] == 2:
                    datos.iloc[i, 1] = 0
                elif datos.loc[i][1] >= med2 and datos.loc[i][5] == 2:
                    datos.iloc[i, 1] = 1
        else:
            for i in range(len(datos)):
                if datos.loc[i][3] < med1 and datos.loc[i][-1] == 1:
                    datos.iloc[i, 3] = 0
                elif datos.loc[i][3] >= med1 and datos.loc[i][-1] == 1:
                    datos.iloc[i, 3] = 1
                elif datos.loc[i][3] < med3 and datos.loc[i][-1] == 3:
                    datos.iloc[i, 3] = 0
                elif datos.loc[i][3] >= med3 and datos.loc[i][-1] == 3:
                    datos.iloc[i, 3] = 1
                elif datos.loc[i][3] < med0 and datos.loc[i][-1] == 0:
                    datos.iloc[i, 3] = 0
                elif datos.loc[i][3] >= med0 and datos.loc[i][-1] == 0:
                    datos.iloc[i, 3] = 1
                elif datos.loc[i][3] < med2 and datos.loc[i][-1] == 2:
                    datos.iloc[i, 3] = 0
                elif datos.loc[i][3] >= med2 and datos.loc[i][-1] == 2:
                    datos.iloc[i, 3] = 1

        df = datos[datos['clase'] == 0]['linfocitos']
        med0 = stats.median(df)
        df = datos[datos['clase'] == 1]['linfocitos']
        med1 = stats.median(df)
        df = datos[datos['clase'] == 2]['linfocitos']
        med2 = stats.median(df)
        df = datos[datos['clase'] == 3]['linfocitos']
        med3 = stats.median(df)

        if datos['linfocitos'][0] == datos.loc[0][2]:
            print("entro-linfocitos")
            for i in range(len(datos)):
                if datos.loc[i][2] < med1 and datos.loc[i][-1] == 1:
                    datos.iloc[i, 2] = 0
                elif datos.loc[i][2] >= med1 and datos.loc[i][-1] == 1:
                    datos.iloc[i, 2] = 1
                elif datos.loc[i][2] < med3 and datos.loc[i][-1] == 3:
                    datos.iloc[i, 2] = 0
                elif datos.loc[i][2] >= med3 and datos.loc[i][-1] == 3:
                    datos.iloc[i, 2] = 1
                elif datos.loc[i][2] < med0 and datos.loc[i][-1] == 0:
                    datos.iloc[i, 2] = 0
                elif datos.loc[i][2] >= med0 and datos.loc[i][-1] == 0:
                    datos.iloc[i, 2] = 1
                elif datos.loc[i][2] < med2 and datos.loc[i][-1] == 2:
                    datos.iloc[i, 2] = 0
                elif datos.loc[i][2] >= med2 and datos.loc[i][-1] == 2:
                    datos.iloc[i, 2] = 1
        else:
            for i in range(len(datos)):
                if datos.loc[i][4] < med1 and datos.loc[i][-1] == 1:
                    datos.iloc[i, 4] = 0
                elif datos.loc[i][4] >= med1 and datos.loc[i][-1] == 1:
                    datos.iloc[i, 4] = 1
                elif datos.loc[i][4] < med3 and datos.loc[i][-1] == 3:
                    datos.iloc[i, 4] = 0
                elif datos.loc[i][4] >= med3 and datos.loc[i][-1] == 3:
                    datos.iloc[i, 4] = 1
                elif datos.loc[i][4] < med0 and datos.loc[i][-1] == 0:
                    datos.iloc[i, 4] = 0
                elif datos.loc[i][4] >= med0 and datos.loc[i][-1] == 0:
                    datos.iloc[i, 4] = 1
                elif datos.loc[i][4] < med2 and datos.loc[i][-1] == 2:
                    datos.iloc[i, 4] = 0
                elif datos.loc[i][4] >= med2 and datos.loc[i][-1] == 2:
                    datos.iloc[i, 4] = 1

        df = datos[datos['clase'] == 0]['hematocritos']
        med0 = stats.median(df)
        df = datos[datos['clase'] == 1]['hematocritos']
        med1 = stats.median(df)
        df = datos[datos['clase'] == 2]['hematocritos']
        med2 = stats.median(df)
        df = datos[datos['clase'] == 3]['hematocritos']
        med3 = stats.median(df)

        if datos['hematocritos'][0] == datos.loc[0][3]:
            print("entro-hematocritos")
            for i in range(len(datos)):
                if datos.loc[i][3] < med1 and datos.loc[i][-1] == 1:
                    datos.iloc[i, 3] = 0
                elif datos.loc[i][3] >= med1 and datos.loc[i][-1] == 1:
                    datos.iloc[i, 3] = 1
                elif datos.loc[i][3] < med3 and datos.loc[i][-1] == 3:
                    datos.iloc[i, 3] = 0
                elif datos.loc[i][3] >= med3 and datos.loc[i][-1] == 3:
                    datos.iloc[i, 3] = 1
                elif datos.loc[i][3] < med0 and datos.loc[i][-1] == 0:
                    datos.iloc[i, 3] = 0
                elif datos.loc[i][3] >= med0 and datos.loc[i][-1] == 0:
                    datos.iloc[i, 3] = 1
                elif datos.loc[i][3] < med2 and datos.loc[i][-1] == 2:
                    datos.iloc[i, 3] = 0
                elif datos.loc[i][3] >= med2 and datos.loc[i][-1] == 2:
                    datos.iloc[i, 3] = 1
        else:
            for i in range(len(datos)):
                if datos.loc[i][5] < med1 and datos.loc[i][-1] == 1:
                    datos.iloc[i, 5] = 0
                elif datos.loc[i][5] >= med1 and datos.loc[i][-1] == 1:
                    datos.iloc[i, 5] = 1
                elif datos.loc[i][5] < med3 and datos.loc[i][-1] == 3:
                    datos.iloc[i, 5] = 0
                elif datos.loc[i][5] >= med3 and datos.loc[i][-1] == 3:
                    datos.iloc[i, 5] = 1
                elif datos.loc[i][5] < med0 and datos.loc[i][-1] == 0:
                    datos.iloc[i, 5] = 0
                elif datos.loc[i][5] >= med0 and datos.loc[i][-1] == 0:
                    datos.iloc[i, 5] = 1
                elif datos.loc[i][5] < med2 and datos.loc[i][-1] == 2:
                    datos.iloc[i, 5] = 0
                elif datos.loc[i][5] >= med2 and datos.loc[i][-1] == 2:
                    datos.iloc[i, 5] = 1

        df = datos[datos['clase'] == 0]['leucocitos']
        med0 = stats.median(df)
        df = datos[datos['clase'] == 1]['leucocitos']
        med1 = stats.median(df)
        df = datos[datos['clase'] == 2]['leucocitos']
        med2 = stats.median(df)
        df = datos[datos['clase'] == 3]['leucocitos']
        med3 = stats.median(df)

        if datos['leucocitos'][0] == datos.loc[0][4]:
            print("entro-leucocitos")
            for i in range(len(datos)):
                if datos.loc[i][4] < med1 and datos.loc[i][-1] == 1:
                    datos.iloc[i, 4] = 0
                elif datos.loc[i][4] >= med1 and datos.loc[i][-1] == 1:
                    datos.iloc[i, 4] = 1
                elif datos.loc[i][4] < med3 and datos.loc[i][-1] == 3:
                    datos.iloc[i, 4] = 0
                elif datos.loc[i][4] >= med3 and datos.loc[i][-1] == 3:
                    datos.iloc[i, 4] = 1
                elif datos.loc[i][4] < med0 and datos.loc[i][-1] == 0:
                    datos.iloc[i, 4] = 0
                elif datos.loc[i][4] >= med0 and datos.loc[i][-1] == 0:
                    datos.iloc[i, 4] = 1
                elif datos.loc[i][4] < med2 and datos.loc[i][-1] == 2:
                    datos.iloc[i, 4] = 0
                elif datos.loc[i][4] >= med2 and datos.loc[i][-1] == 2:
                    datos.iloc[i, 4] = 1
        else:
            for i in range(len(datos)):
                if datos.loc[i][6] < med1 and datos.loc[i][-1] == 1:
                    datos.iloc[i, 6] = 0
                elif datos.loc[i][6] >= med1 and datos.loc[i][-1] == 1:
                    datos.iloc[i, 6] = 1
                elif datos.loc[i][6] < med3 and datos.loc[i][-1] == 3:
                    datos.iloc[i, 6] = 0
                elif datos.loc[i][6] >= med3 and datos.loc[i][-1] == 3:
                    datos.iloc[i, 6] = 1
                elif datos.loc[i][6] < med0 and datos.loc[i][-1] == 0:
                    datos.iloc[i, 6] = 0
                elif datos.loc[i][6] >= med0 and datos.loc[i][-1] == 0:
                    datos.iloc[i, 6] = 1
                elif datos.loc[i][6] < med2 and datos.loc[i][-1] == 2:
                    datos.iloc[i, 6] = 0
                elif datos.loc[i][6] >= med2 and datos.loc[i][-1] == 2:
                    datos.iloc[i, 6] = 1

        """
    X = datos.loc[:, datos.columns != 'clase']
    Y = datos['clase']

    return X, Y


# Funcion para particionar los datasets
def split_data(n_groups, X, y):
    groups_X = []
    groups_Y = []

    kf = KFold(n_splits=n_groups, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        groups_X.append(X_train)
        groups_Y.append(y_train)

    return np.array(groups_X), np.array(groups_Y)


# Funcion que imprime cada iteracion
def print_data(a, b, c, d, e, i, F1):
    print(str(a) + " " + str(b) + " " + str(c) + " " + str(d) + " " + str(e) + " " + str(i)
          + " " + str(F1))


# Funcion que devuelve el F1-score
def F1(model, x, y):
    predicted = model.predict(x)
    results = f1_score(y,predicted, average=None)
    acum = 0
    total = len(results)
    for i in results:
        acum = acum + i
    return acum / total

def F1test(y, predicted):
    results = f1_score(y,predicted, average=None)
    acum = 0
    total = len(results)
    cont =1
    for i in results:
        print("F1",cont,":", i)
        acum = acum + i
        cont = cont +1
    print("F1 total: ",acum / total)



# Funcion de normalizacion
def normalizacion(datos):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(datos)
    datos = pd.DataFrame(scaled)
    return datos

def print_evaluation(y,predicted):
    matriz =confusion_matrix(y, predicted)
    seaborn.heatmap(matriz, cmap='inferno',cbar=False,annot=True,fmt="")
    plt.title("Matriz Confusion")
    plt.show()
    return matriz

