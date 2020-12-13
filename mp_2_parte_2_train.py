import utilities as util
import sys
from sklearn.ensemble import RandomForestClassifier as model_gen
import pickle

def main():
    
    data_set = sys.argv[1]
    model_name = sys.argv[2]
    
    trees = 30
    criterion = 'entropy'
    max_depth = None
    max_features = 'log2'    
    
    X,Y = util.process_data(data_set)
    
    model = model_gen(
        n_estimators = trees,
        criterion = criterion,
        max_depth = max_depth,
        max_features = max_features
    )

    model.fit(X,Y)                            
    with open(model_name, "wb") as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    main()
    

