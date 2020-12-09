from sklearn.metrics import f1_score

def f1(model,x,y):
    predicted = model.predict(x)
    f1=f1_score(predicted,y)
    return f1


