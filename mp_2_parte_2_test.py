import sys
import pickle
import utilities as util

def efectividad(matriz,Y):
    total=matriz[0][0]+matriz[1][1]+matriz[2][2]+matriz[3][3]
    return total/(len(Y))


def main():

    data_set = sys.argv[1]
    model_name = sys.argv[2]

                     
    with open(model_name, "rb") as f:
        model = pickle.load(f)

    X,Y = util.process_data(data_set)


    Y_predic = model.predict(X)

    matriz=util.print_evaluation(Y,Y_predic)
    util.F1test(Y,Y_predic)
    print("Efectividad: ",(str)(efectividad(matriz,Y)*100)+"%")





if __name__ == '__main__':
    main()
    