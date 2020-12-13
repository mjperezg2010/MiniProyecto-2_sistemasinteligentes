
from sklearn.naive_bayes import GaussianNB as model1
from sklearn.naive_bayes import BernoulliNB as model2
from sklearn.naive_bayes import CategoricalNB as model3
import utilities as util
import numpy as np

def main():
    
    data_sets = ["clinica_train_synth_dengue.csv",
                    "laboratorio_train_synth_dengue.csv",
                    "completo_train_synth_dengue.csv"]
                
    groups = 5

    for a in data_sets:
                
        for m in [0,1,2]:
            print(m)
            if m == 0:
                X,Y = util.process_data(a)
                #X = util.normalizacion(X)
                model = model1(
                    
                )
            elif m == 1:
                X,Y = util.process_data(a)
                X = util.preprocesar2(X)
                model = model2(
                    
                )
            else:
                X,Y = util.process_data(a)
                X = util.preprocesar2(X)
                model = model3(
                    
                )                                
            groups_X, groups_Y = util.split_data(groups, X, Y)
            groups_Y = groups_Y.astype('int')
            for i in range(groups):
                validation_X = groups_X[i]
                validation_Y = groups_Y[i]
                if i != 0:
                    training_X = groups_X[0]
                    training_Y = groups_Y[0]
                else:
                    training_X = groups_X[1]
                    training_Y = groups_Y[1]
                for j in range(1, groups):
                    if j != i:
                        training_X = np.concatenate((training_X, groups_X[i]))
                        training_Y = np.concatenate((training_Y, groups_Y[i]))                
                model.fit(training_X, training_Y)
                F1_val = util.F1(model, validation_X, validation_Y)
                print(F1_val)
                    

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

