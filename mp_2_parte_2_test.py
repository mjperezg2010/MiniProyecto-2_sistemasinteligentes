
import pickle
import sys
import pickle
import utilities as util

def main():

    data_set = sys.argv[1]
    model_name = sys.argv[2]

                     
    with open(model_name, "rb") as f:
        model = pickle.load(f)

    X,Y = util.process_data(data_set)


    Y_predic = model(X)

    util.print_evaluation(Y,Y_predic)


if __name__ == '__main__':
    main()
    