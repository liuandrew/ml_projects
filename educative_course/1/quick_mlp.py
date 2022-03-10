import torch
from torch import nn
from torch.optim import Adam
from vscode import load_data
import numpy as np
import matplotlib.pyplot as plt


X_train, X_test, y_train, y_test = load_data()
input_dim = X_train.shape[1]
output_dim = len(np.unique(y_test))

class Clasifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super().__init__()
        # self.layers = nn.Linear(input_dim, output_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        # sig = nn.Sigmoid()
        self.softmax = nn.Softmax()
    
    def forward(self, X):
        logits = self.layers(X)
        probs = self.softmax(logits)
        return probs
        
clf = Clasifier(input_dim, output_dim)
optimizer = Adam(clf.parameters())
loss_fun = nn.CrossEntropyLoss()
X = torch.tensor(X_train, dtype=torch.float, requires_grad=True)
y = torch.tensor(y_train, dtype=torch.int64)

losses = []
for i in range(1000):
    probs = clf.forward(X)
    loss = loss_fun(probs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
plt.plot(np.arange(len(losses)), losses)
plt.show()
    
# test
X_test = torch.tensor(X_test, dtype=torch.float, requires_grad=False)
y_test = torch.tensor(y_test, dtype=torch.int64)
y_pred = clf.forward(X_test)
y_pred_labels = y_pred.argmax(axis=1)