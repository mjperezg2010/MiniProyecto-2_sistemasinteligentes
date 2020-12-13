import sys
import pickle
import utilities as util

def efectividad(matriz,Y):
    total=matriz[0][0]+matriz[1][1]+matriz[2][2]+matriz[3][3]
    return total/(len(Y))


def main():

    data_set = sys.argv[1]
    type_model = sys.argv[2]
    model_name = sys.argv[3]

                     
    with open(model_name, "rb") as f:
        model = pickle.load(f)

    if type_model == 'gaussian':        
        X,Y = util.process_data(data_set)
    elif type_model == 'bernoulli':        
        X,Y = util.preprocesar2(data_set)
    elif type_model == 'categorical':        
        X,Y = util.preprocesar2(data_set)
        X = X+1
    else:
        print("esta malo xdXxdXd")


    Y_predic = model.predict(X)

    matriz=util.print_evaluation(Y,Y_predic)
    util.F1test(Y,Y_predic)
    print("Efectividad: ",(str)(efectividad(matriz,Y)*100)+"%")


if __name__ == '__main__':
    main()
    