import utilities as util
from sklearn.svm import SVC as model_svm
import numpy as np
import sys


def main():
            
    data_sets = [sys.argv[1]]
    
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
                    values_f1 = 0
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
                        values_f1 += util.F1(model, validation_X, validation_Y)
                    print("dataset: "+a)
                    print("kernel: "+kernel)
                    print("C: "+str(c))
                    print("gamma: "+str(g))                    
                    print("F1: "+str(values_f1/groups))
                            


if __name__ == '__main__':
    main()

