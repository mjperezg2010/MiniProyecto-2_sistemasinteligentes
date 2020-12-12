import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

#Funciones

#Funcion para cargar el archivo
def load_file(file_name):
    return pd.read_csv(file_name)

#Funcion de procesamiento de los datos
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
    string_array=string_array.replace("Dengue_Grave",0,regex=True)
    string_array = string_array.replace("Dengue_NoGrave_NoSignos", 1, regex=True)
    string_array = string_array.replace("Dengue_NoGrave_SignosAlarma", 2, regex=True)
    string_array = string_array.replace("No_Dengue", 3, regex=True)

    X = string_array.loc[:, string_array.columns != 'clase']
    Y = string_array['clase']
    return X,Y

def preprocesar2(datos):
    if datos['plaquetas'][0] == datos.loc[0][1]:
        for i in range(len(datos)):
            if datos.loc[i][1] < 193000 and datos.loc[i][-1] == 1:
                datos.iloc[[i, 1]] = 0
            elif datos.loc[i][1] >= 193000 and datos.loc[i][-1] == 1:
                datos.iloc[[i, 1]] = 1
            elif datos.loc[i][1] < 275000 and datos.loc[i][-1] == 3:
                datos.iloc[[i, 1]] = 0
            elif datos.loc[i][1] >= 275000 and datos.loc[i][-1] == 3:
                datos.iloc[[i, 1]] = 1
            elif datos.loc[i][1] < 125000 and datos.loc[i][-1] == 0:
                datos.iloc[[i, 1]] = 0
            elif datos.loc[i][1] >= 125000 and datos.loc[i][-1] == 0:
                datos.iloc[[i, 1]] = 1
            elif datos.loc[i][1] < 198000 and datos.loc[i][-1] == 2:
                datos.iloc[[i, 1]] = 0
            elif datos.loc[i][1] >= 198000 and datos.loc[i][-1] == 2:
                datos.iloc[[i, 1]] = 1
    else:
        for i in range(len(datos)):
            if datos.loc[i][3] < 193000 and datos.loc[i][-1] == 1:
                datos.iloc[[i, 3]] = 0
            elif datos.loc[i][3] >= 193000 and datos.loc[i][-1] == 1:
                datos.iloc[[i, 3]] = 1
            elif datos.loc[i][3] < 275000 and datos.loc[i][-1] == 3:
                datos.iloc[[i, 3]] = 0
            elif datos.loc[i][3] >= 275000 and datos.loc[i][-1] == 3:
                datos.iloc[[i, 3]] = 1
            elif datos.loc[i][3] < 125000 and datos.loc[i][-1] == 0:
                datos.iloc[[i, 3]] = 0
            elif datos.loc[i][3] >= 125000 and datos.loc[i][-1] == 0:
                datos.iloc[[i, 3]] = 1
            elif datos.loc[i][3] < 198000 and datos.loc[i][-1] == 2:
                datos.iloc[[i, 3]] = 0
            elif datos.loc[i][3] >= 198000 and datos.loc[i][-1] == 2:
                datos.iloc[[i, 3]] = 1

    if datos['linfocitos'][0] == datos.loc[0][2]:
        for i in range(len(datos)):
            if datos.loc[i][2] < 0.41 and datos.loc[i][-1] == 1:
                datos.iloc[[i, 2]] = 0
            elif datos.loc[i][2] >= 0.41 and datos.loc[i][-1] == 1:
                datos.iloc[[i, 2]] = 1
            elif datos.loc[i][2] < 0.175 and datos.loc[i][-1] == 3:
                datos.iloc[[i, 2]] = 0
            elif datos.loc[i][2] >= 0.175 and datos.loc[i][-1] == 3:
                datos.iloc[[i, 2]] = 1
            elif datos.loc[i][2] < 0.47 and datos.loc[i][-1] == 0:
                datos.iloc[[i, 2]] = 0
            elif datos.loc[i][2] >= 0.47 and datos.loc[i][-1] == 0:
                datos.iloc[[i, 2]] = 1
            elif datos.loc[i][2] < 0.405 and datos.loc[i][-1] == 2:
                datos.iloc[[i, 2]] = 0
            elif datos.loc[i][2] >= 0.405 and datos.loc[i][-1] == 2:
                datos.iloc[[i, 2]] = 1
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
                datos.iloc[[i, 4]] = 0
            elif datos.loc[i][4] >= 0.405 and datos.loc[i][-1] == 2:
                datos.iloc[[i, 4]] = 1

    if datos['hematocritos'][0] == datos.loc[0][3]:
        for i in range(len(datos)):
            if datos.loc[i][3] < 0.44 and datos.loc[i][-1] == 1:
                datos.iloc[[i, 3]] = 0
            elif datos.loc[i][3] >= 0.44 and datos.loc[i][-1] == 1:
                datos.iloc[[i, 3]] = 1
            elif datos.loc[i][3] < 0.422 and datos.loc[i][-1] == 3:
                datos.iloc[[i, 3]] = 0
            elif datos.loc[i][3] >= 0.422 and datos.loc[i][-1] == 3:
                datos.iloc[[i, 3]] = 1
            elif datos.loc[i][3] < 0.452 and datos.loc[i][-1] == 0:
                datos.iloc[[i, 3]] = 0
            elif datos.loc[i][3] >= 0.452 and datos.loc[i][-1] == 0:
                datos.iloc[[i, 3]] = 1
            elif datos.loc[i][3] < 0.43 and datos.loc[i][-1] == 2:
                datos.iloc[[i, 3]] = 0
            elif datos.loc[i][3] >= 0.43 and datos.loc[i][-1] == 2:
                datos.iloc[[i, 3]] = 1
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
                datos.iloc[[i, 5]] = 0
            elif datos.loc[i][5] >= 0.405 and datos.loc[i][-1] == 2:
                datos.iloc[[i, 5]] = 1

    if datos['leucocitos'][0] == datos.loc[0][3]:
        for i in range(len(datos)):
            if datos.loc[i][4] < 6100 and datos.loc[i][-1] == 1:
                datos.iloc[[i, 4]] = 0
            elif datos.loc[i][4] >= 6100 and datos.loc[i][-1] == 1:
                datos.iloc[[i, 4]] = 1
            elif datos.loc[i][4] < 8250 and datos.loc[i][-1] == 3:
                datos.iloc[[i, 4]] = 0
            elif datos.loc[i][4] >= 8250 and datos.loc[i][-1] == 3:
                datos.iloc[[i, 4]] = 1
            elif datos.loc[i][4] < 5750 and datos.loc[i][-1] == 0:
                datos.iloc[[i, 4]] = 0
            elif datos.loc[i][4] >= 5750 and datos.loc[i][-1] == 0:
                datos.iloc[[i, 4]] = 1
            elif datos.loc[i][4] < 6250 and datos.loc[i][-1] == 2:
                datos.iloc[[i, 4]] = 0
            elif datos.loc[i][4] >= 6250 and datos.loc[i][-1] == 2:
                datos.iloc[[i, 4]] = 1
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

"""
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
            actual_X.append(X.values[temp[i*int(len(X.values)/n_groups)+j]])
            actual_Y.append(Y.values[temp[i*int(len(X.values)/n_groups)+j]])
        groups_X.append(actual_X)
        groups_Y.append(actual_Y)
    return groups_X,groups_Y
"""

#Funcion para particionar los datasets
def split_data(n_groups,X,y):
    groups_X = []
    groups_Y = []

    kf = KFold(n_splits=n_groups, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        groups_X.append(X_train)
        groups_Y.append(y_train)

    return np.array(groups_X) , np.array(groups_Y)

#Funcion que imprime cada iteracion
def print_data(a, b, c, d, e, i, F1):
    print(str(a)+" "+str(b)+" "+str(c)+" "+str(d)+" "+str(e)+" "+str(i)
        +" "+str(F1))

#Funcion que devuelve el F1-score
def F1(model,x,y):
    predicted = model.predict(x)
    results = f1_score(predicted, y, average=None)
    acum = 0
    total = len(results)
    for i in results:
        acum = acum + i
    return acum / total

#Funcion de normalizacion
def normalizacion(datos):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(datos)
    datos = pd.DataFrame(scaled)
    return datos