# import unittest

from vscode import load_data

# class TestLogisticRegression(unittest.TestCase):
#     def test_load_data(self):
#         data, target = load_data()
#         assert data.shape == (1797, 64)

def test_load_data():
    data, target = load_data(split = False)
    assert data.shape == (1797, 64) 
    
    X_train, X_test, y_train, y_test = load_data(split = True)
    assert X_train.shape == (1347, 64)