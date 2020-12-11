
import utilities as util
from sklearn.svm import SVC as model_svm

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

def main():
    
    data_sets = ["clinica_train_synth_dengue.csv",
                    "laboratorio_train_synth_dengue.csv",
                    "completo_train_synth_dengue.csv"]
    
    C = [ i/10 for i in range(1,15) ]
    kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    gamma = ['scale', 'auto']+[ i/10 for i in range(1,15) ]
    
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
                    groups_X,groups_Y = util.split_data(groups,X,Y)
                    for i in range(groups):
                        validation_X = groups_X[i]
                        validation_Y = groups_Y[i]
                        for j in range(groups):
                            if j != i:
                                training_X += groups_X[i]
                                training_Y += groups_Y[i]                            
                        model.fit(training_X,training_Y)
                        F1_val = util.F1(model,validation_X,validation_Y)
                        print(a+" "+c+" "+kernel+" "+g)
                            

    

