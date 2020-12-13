import utilities as util
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
import pickle

def main():
    
    data_set = sys.argv[1]
    type_model = sys.argv[2]
    model_name = sys.argv[3]
            
       

    if type_model == 'gaussian':
        model = GaussianNB()
        X,Y = util.process_data(data_set)
    elif type_model == 'bernoulli':
        model = BernoulliNB()
        X,Y = util.preprocesar2(data_set)
    elif type_model == 'categorical':
        model = CategoricalNB()
        X,Y = util.preprocesar2(data_set)
        X=X+1
    else:
        print("esta malo xdXxdXd")
    
    model.fit(X,Y)        
    with open(model_name, "wb") as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    main()
    

