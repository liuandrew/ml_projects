from distutils.log import Log
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

import numpy as np
from typing import Tuple

def load_data(filename: str = None, split: bool = True) -> Tuple [np.ndarray, np.ndarray]:
    data = load_digits()
    X = data['data']
    y = data['target']
    
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test
    else:
        return X, y
# X_train = X[:1500]
# y_train = y[:1500]

X_train, X_test, y_train, y_test = load_data()
clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

a = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)

data = {
    'a': 5,
    'b': [[5, 10],
          [15, 20],
          [25, 30]],
    'c': 'sixty five'
}
