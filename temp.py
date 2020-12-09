from sklearn.metrics import f1_score

def f1(model,x,y):
    predicted = model.predict(x)
    results= f1_score(predicted, y, average=None)
    acum=0
    total=len(results)
    for i in results:
        acum=acum+i

    return acum/total



