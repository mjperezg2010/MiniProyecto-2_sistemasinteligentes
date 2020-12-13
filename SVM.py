
import utilities as util
from sklearn.svm import SVC as model_svm
import numpy as np



def main():
    
    #data_sets = ["clinica_train_synth_dengue.csv",
    #                "laboratorio_train_synth_dengue.csv",
    #                "completo_train_synth_dengue.csv"]
    data_sets=["clinica_train_synth_dengue.csv"]
    C = [ 0.5,1,1.5 ]
    kernels = ['linear', 'rbf', 'sigmoid']
    gamma = ['scale', 'auto']
    
    groups = 5

    for a in data_sets:
        X,Y = util.process_data(a)
        X = util.normalizacion(X)
        for c in C:
            for kernel in kernels:
                for g in gamma:                    
                    model = model_svm(
                        C = c,
                        kernel = kernel,
                        gamma = g
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
                        print(a,c,kernel,g,F1_val)
                            

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

