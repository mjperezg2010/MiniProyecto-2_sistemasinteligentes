import utilities as util
import sys
from sklearn.ensemble import RandomForestClassifier as model_gen
import numpy as np

def main():
    
    data_sets = [sys.argv[1]]
    
    trees = [i*10 for i in range(1,5)]    
    criterion = ['gini','entropy']    
    max_depth = [None,3,6]
    max_features = ['auto','log2']

    groups = 5

    for a in data_sets:
        X,Y = util.process_data(a)
        for b in trees:
            for c in criterion:
                for d in max_depth:
                    for e in max_features:                 
                        model = model_gen(
                            n_estimators = b,
                            criterion = c,
                            max_depth = d,
                            max_features = e
                        )
                        groups_X,groups_Y = util.split_data(groups,X,Y)
                        values_f1 = ""
                        groups_Y=groups_Y.astype('int')
                        values_f1 = 0
                        for i in range(groups):
                            validation_X = groups_X[i]
                            validation_Y = groups_Y[i]
                            if i!=0:
                                training_X=groups_X[0]
                                training_Y=groups_Y[0]
                            else:
                                training_X = groups_X[1]
                                training_Y = groups_Y[1]

                            for j in range(1,groups):
                                if j != i:
                                    training_X=np.concatenate((training_X,groups_X[i]))
                                    training_Y = np.concatenate((training_Y, groups_Y[i]))

                            model.fit(training_X,training_Y)                            
                            F1_val = util.F1(model,validation_X,validation_Y)
                            values_f1 += F1_val
                        print("Dataset: "+data_sets[0])
                        print("num arboles: "+str(b))
                        print("criterio: "+c)
                        print("profundidad: "+str(d))
                        print("features: "+str(e))
                        print("F1: "+str(values_f1/groups)) 
                        print()                                           
                            
if __name__ == '__main__':
    main()
    

