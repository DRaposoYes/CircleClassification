import torch
from torch import nn # nn contains all building blocks for neural networks
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sklearn
from sklearn.datasets import make_circles
import pandas as pd

## toy data set
## 1 data

n_samples = 1000

## create circles

print("Create circles, will be made in numpy type")
print(">")

X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)


print(len(X), len(y))

print(f"First 5 samples of X: {X[:5]}")
print(f"First 5 samples of y: {y[:5]}")


## make dataframe
print("make dataframe")
print(">")

circles = pd.DataFrame({"X1": X[:, 0], 
                        "X2": X[:, 1],
                        "label": y})


print(circles.head(10))


##1.1  Visualize
print("visualize")
print(">")

plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
plt.title("Data of 2 circles")
# plt.show()

print("shapes")
print(">")

print(X.shape, y.shape)

## view the first example of features and lables

print("view the first example of features and lables")
print(">")

X_sample = X[0]
y_sample = y[0]

##1.2 turn data into tensors

print("turn data into tensors")
print(">")

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)


print(X[:5], y[:5])


## Split data into training and test sets

print("Split data into training and test sets")
print(">")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, #20% test data, rest is training
                                                    random_state=42
                                                    )

print(len(X_train), len(X_test), len(y_train), len(y_test) ,n_samples)




## 2 building a model
print("building a model")
print(">")

## device agnistic code:

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

## complex way
class CircleModelV0(nn.Module):
        def __init__(self):
            super().__init__()

            self.layer_1 = nn.Linear(in_features=2, out_features=5)
            self.layer_2 = nn.Linear(in_features=5, out_features=1)

        def forward(self, x):
              return self.layer_2(self.layer_1(x))
    

## old model, keep here for reference
#Cmodel = CircleModelV0().to(device)
#print(Cmodel)


## check if cuda working
#print(device)
#print(next(Cmodel.parameters()).device)


# replicate model above

print("replicate model above differently, model 0, linear model without more layers or more features")
print(">")

## it does what Class CircleModelV0 does. but behind the scenes with less code
model_0 = nn.Sequential(
      nn.Linear(in_features=2, out_features=5),
      nn.Linear(in_features=5, out_features=1)
).to(device)

print(model_0)


print("state model_0 dict")
print(">")

print(model_0.state_dict())

## Make predictions
print("untrained predictions")
print(">")

with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))

print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(X_test)}, Shape: {X_test.shape}")
print(f"\nFirst 10 predictions: \n{untrained_preds[:10]}")
print(f"\nFirst 10 lables: \n{y_test[:10]}")

## setup loss function and optimizer

print("setup loss function and optimizer, calculate accuracy function")
print(">")

## setup loss

loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogisticsLoss = sigmoid activation function

#optimizer

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

#calculate accuracy

def acurracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

## train model

print("train model / train steps")
print(">")



## logits / Raw Guesses
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]

## sigmoid function on model logits, otherwise you cant do torch.round

y_pred_probs = torch.sigmoid(y_logits)

## find predicted lables

y_preds = torch.round(y_pred_probs)

## in full. logits (raw guess) > pred prob (example 0.76 chance) > pred lable (example: 1) 
y_pred_lables = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))



## Check for equality

print(torch.eq(y_preds.squeeze(), y_pred_lables.squeeze()))

## get rid of extra dimention
y_preds.squeeze()



## building a training and testing loop

print("building training loop")
print(">")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 11

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

## looping training

print("looping model 0 training")
print(">")

for epoch in range(epochs):
     #training
    model_0.train()

    ## forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # Turn logits (Raw guesses) -> pred probs -> pred lables (result)

    ## calculate loss

    loss = loss_fn(y_logits, # nn.BCEWlogits expects raw logits
                   y_train)
    
    acc = acurracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
    ## optimizer zero grad
    
    optimizer.zero_grad()

    ## loss backward
    loss.backward()

    ## optimizer step
    optimizer.step()

    #testing
    model_0.eval()
    with torch.inference_mode():
         # forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        #calculate test loss
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = acurracy_fn(y_true=y_test,
                               y_pred=test_pred)
    if epoch % 10 == 0:
         print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test acc: {test_acc}%")
                            
             
# print(circles.label.value_counts())


## visualize predictions (Model currently isnt training correctly wheen first writing this)

import requests 
from pathlib import Path

if Path("helper_functions.py").is_file():
     print("helper_functions.py already exists")
else:
     print("downloading helper_functions.py...")
     request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
     with open("helper_functions.py", "wb") as f:
          f.write(request.content)
from helper_functions import plot_predictions, plot_decision_boundary


## Plot decision boundry
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train, Model_0")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test, Model_0")
plot_decision_boundary(model_0, X_test, y_test)
#plt.show()



print("training model 1, linear model WITH more layers and features")
print(">")


class CircleModelV1(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=2, out_features=10)
            self.layer_2 = nn.Linear(in_features=10, out_features=10)
            self.layer_3 = nn.Linear(in_features=10, out_features=1)

        def forward(self, x):
              z = self.layer_1(x)
              z = self.layer_2(z)
              z = self.layer_3(z)
              return self.layer_3(self.layer_2(self.layer_1(x)))


model_1 = CircleModelV1().to(device)


##loss function for model 1
loss_fn = nn.BCEWithLogitsLoss()

## Create optimizer for model 1
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 11

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_train.to(device), y_train.to(device)

for epoch in range(epochs):
     model_1.train

     y_logits = model_1(X_train).squeeze()
     y_pred = torch.round(torch.sigmoid(y_logits))


     loss = loss_fn(y_logits, y_train)
     acc = acurracy_fn(y_true=y_train,
                       y_pred=y_pred)
     
     optimizer.zero_grad()

     loss.backward()

     optimizer.step()

     model_1.eval()
     with torch.inference_mode():
          test_logits = model_1(X_test).squeeze()
          test_pred = torch.round(torch.sigmoid(test_logits))

          test_loss = loss_fn(test_logits,
                              y_test)
          test_acc = acurracy_fn(y_true=y_test,
                                 y_pred=test_pred)


     if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test acc: {test_acc}%")


## plt boundary model 1
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train, Model_1")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test, Model_1")
plot_decision_boundary(model_1, X_test, y_test)
# plt.show()



