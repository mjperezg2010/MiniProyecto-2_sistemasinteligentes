import utilities as util
from sklearn.svm import SVC as model_svm
import numpy as np
import sys


def main():
    
    data_set = sys.argv[1]
    model_name = sys.argv[2]

    c = 1.5
    kernel = 'rbf'
    gamma = 'scale'

    X,Y = util.process_data(data_set)
    X = util.normalizacion(X)

    model = model_svm(
        C = C,
        kernel = kernel,
        gamma = gamma
    )

    model.fit(X,Y)
    with open(model_name, "wb") as f:
        pickle.dump(model, f)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()