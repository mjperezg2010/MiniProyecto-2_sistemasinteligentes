from sklearn.naive_bayes import GaussianNB as model1
from sklearn.naive_bayes import BernoulliNB as model2
from sklearn.naive_bayes import CategoricalNB as model3
import utilities as util
import numpy as np
import sys

def main():
    
    data_sets = [sys.argv[1]]
            
    groups = 5

    for a in data_sets:
        print()
        for m in [0,1,2]:
            print(m)
            if m == 0:
                print("Gaussian")
                X,Y = util.process_data(a)

                model = model1(
                    
                )
            elif m == 1:
                print("Bernoulli")

                X,Y = util.preprocesar2(a)
                model = model2(
                    
                )
            else:
                print("Categorical")

                X,Y = util.preprocesar2(a)
                X = X +10
                model = model3(
                    
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
            print("F1: "+str(values_f1/groups))            


if __name__ == '__main__':
    main()

