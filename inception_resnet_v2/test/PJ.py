import numpy as np
from sklearn.metrics import confusion_matrix


# 混淆矩阵


def my_classification_report(y_true, y_pred):
    from sklearn.metrics import classification_report
    #labels = list(set(y_true))
    print("classification_report(left: labels):")
    print((classification_report(y_true, y_pred)))
    a = classification_report(y_true, y_pred)
    with open('classification_report_20180803.txt','w') as f:
          f.write(a)



y_true = []
y_pred = []
path = './new.txt'
test = open(path).readlines()
test = np.array([w.strip().split(';')[:2] for w in test])

y_pred = test[:, 1]
y_true = test[:, 0]

my_classification_report(y_true, y_pred)

