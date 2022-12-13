import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

import torchvision.models as models

np.random.seed(42)

#create slightly randomized coordinate points
x = np.random.rand(100, 1)
y = 1 + 2 * x + .1 * np.random.randn(100, 1)

idx = np.arange(100)
np.random.shuffle(idx)

train_idx = idx[:80]
#Validation indexes
val_idx = idx[80:]

x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

device = 'cpu'

x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)
x_val_tensor = torch.from_numpy(x_val).float().to(device)
y_val_tensor = torch.from_numpy(y_val).float().to(device)



class Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)
    
    def forward(self, x):
        return self.linear(x)

#model = Regression().to(device)

model = nn.Sequential(
    nn.Linear(1,1)
).to(device)

lr = 1e-1
epochs = 1000

optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.MSELoss(reduction='mean')

for _ in range(epochs):
    model.train()
    #yhat = a + b * x_train_tensor
    yhat = model(x_train_tensor)

    loss_fn(y_train_tensor, yhat).backward()

    optimizer.step()
    optimizer.zero_grad()

model.eval()
yhat = model(x_val_tensor)
print(loss_fn(y_val_tensor, yhat))

torch.save(model.state_dict(), 'model_weights.pth')