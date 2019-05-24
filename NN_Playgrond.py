#!/usr/bin/env python
# coding: utf-8
# original code for make_moons from
#https://www.datahubbs.com/deep-learning-101-first-neural-network-with-pytorch/
#Modified by Tom Lasinski for the make_circles problem. Also added "animation" to show optimizer fitting the training dara.

import numpy as np
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from celluloid import Camera
import random as R

def boundary(X): # determine boundary between different colored dots
	x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
	y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
	spacing = min(x_max - x_min, y_max - y_min) / 100
	XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),np.arange(y_min, y_max, spacing))
	data = np.hstack((XX.ravel().reshape(-1,1),YY.ravel().reshape(-1,1)))
	data_t = torch.FloatTensor(data)
	db_prob = net(data_t)
	clf = np.where(db_prob<0.5,0,1)
	Z = clf.reshape(XX.shape)
	return(plt.contourf(XX, YY, Z, cmap=plt.cm.Accent, alpha=0.6))

def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    R.seed(101)
    f= R.uniform(1.0-noise/2,1+noise/2)
    n_points =  n_points//2
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360 *f
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * f
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * f
    #n_points2 = int(n_points//2)
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))


print("Using PyTorch Version %s" %torch.__version__)
np.random.seed(0)
torch.manual_seed(0)

type = input('c:for circles, s:for spirals, m:for moons \n')
#print(type)
if type == 'c':
    X, Y = make_circles(n_samples=2000, noise=0.15, factor = 0.1, random_state=1)
    iters = 200
    plot_iter = 100
    plot_now = 1
    print(' circles for {} iteraations'.format(iters))
elif type == 's':
    X,Y = twospirals(2000, 0.2)
    iters = 1000
    plot_iter = 500
    plot_now = 4
    print(' spirals for {} iteraations'.format(iters))
elif type == 'm':
    X, Y = make_moons(2000, noise=0.11)
    iters = 400
    plot_iter = 200
    plot_now = 2
    print(' moons for {} iteraations'.format(iters))
else:
    print('type not found')
    exit()


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.75, random_state=73)
#X, Y = make_moons(2000, noise=0.11)
#X, Y = make_circles(n_samples=2000, noise=0.15, factor = 0.1, random_state=1)
#X,Y = twospirals(2000, 0.2)

fig, ax = plt.subplots(1, 2, figsize=(6,4))
ax[0].set_title('Total Circles Data')
ax[0].scatter(X[:,0], X[:,1], c=Y)

ax[1].set_title('Training Circles Data')
ax[1].scatter(X_train[:,0], X_train[:,1], c=Y_train)
#print(' train shape', X_train.shape, Y_train.shape)
plt.tight_layout()
#plt.show()
plt.show(block=False)
plt.pause(2.0)
plt.close()

# Define network dimensions
n_input_dim = X_train.shape[1]
# Layer size
n_hidden2 = 40
n_hidden = 20 # Number of hidden nodes
n_output = 1 # Number of output nodes = for binary classifier

# Build your network
net = nn.Sequential(
    nn.Linear(n_input_dim, n_hidden),
    nn.ELU(),
    nn.Linear(n_hidden, n_hidden2),
    nn.ELU(),
    nn.Linear(n_hidden2, n_output), nn.Sigmoid()
   )

print(net)

x = torch.FloatTensor([1, 1])
net(x)

#net.cuda()
#x = torch.FloatTensor([1, 1]).to(device='cuda')
#net(x)

loss_func = nn.BCELoss()
learning_rate = 0.005
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

train_loss = []
train_accuracy = []
#iters = 700
print('Shapes: ', Y_train.shape, Y_train.reshape(-1,1).shape)
Y_train_t = torch.FloatTensor(Y_train).reshape(-1, 1) # torch Mickey

X_train_t = torch.FloatTensor(X_train)
print('Shapes: ',X_train_t.shape, Y_train_t.shape)
for i in range(iters):
#    X_train_t = torch.FloatTensor(X_train)
    y_hat = net(X_train_t)    #this is how you make predictions.
    y_hat_round = y_hat.round()

    loss = loss_func(y_hat, Y_train_t)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    y_hat_class = np.where(y_hat.detach().numpy()<0.5, 0, 1)
    accuracy = np.sum(Y_train.reshape(-1,1)==y_hat_class) / len(Y_train)
    train_accuracy.append(accuracy)
    train_loss.append(loss.item())

    if i%plot_now == 0:
    	print(' Iteration: ', i, ' loss: {:.4f}'.format(loss.item()),' accuracy:', accuracy)
    	y_hat_final =  y_hat_round.detach().numpy().reshape(500)  # torch Mickey
    	string = 'Iteration = {}'.format(i)
    	plt.figure(figsize=(5,5))
    	plt.title(string)
    	_ = boundary(X)
    	plt.scatter(X_train [:,0], X_train [:,1], c=y_hat_final) #y_hat_class

    	plt.show(block=False)
    	plt.pause(0.2)
    	plt.close()
    	if i > plot_iter : plot_now = 100

X_test_t = torch.FloatTensor(X_test)
Y_test_t = torch.FloatTensor(Y_test).reshape(-1, 1)
y_hat = net(X_test_t)
loss_test =loss_func(y_hat,Y_test_t)
print('Loss for Test Set: {:.4f}'.format(loss_test.item()))
fig, ax = plt.subplots(2, 1, figsize=(6,6))
ax[0].plot(train_loss)
ax[0].set_ylabel('Loss')
ax[0].set_title('Training Loss')

ax[1].plot(train_accuracy)
ax[1].set_ylabel('Classification Accuracy')
ax[1].set_title('Training Accuracy')

plt.tight_layout()
plt.show()

# Pass test data
X_test_t = torch.FloatTensor(X_test)
y_hat_test = net(X_test_t)
y_hat_test_class = np.where(y_hat_test.detach().numpy()<0.5, 0, 1)
test_accuracy = np.sum(Y_test.reshape(-1,1)==y_hat_test_class) / len(Y_test)
print("Test Accuracy {:.3f}".format(test_accuracy))

y_hat_round = y_hat_test.round()
y_hat_final = y_hat_round.detach().numpy().reshape(1500) # torch Mickey

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Training Accuracy:{:.3f}'.format(accuracy))

_ = boundary(X_train)
plt.scatter(X_train[:,0], X_train[:,1], c=Y_train,cmap=plt.cm.Accent)

plt.subplot(1,2,2)
plt.title('Prediction Test Set: {:.3f} Loss'.format(loss_test))

_ = boundary(X_test)
plt.scatter(X_test[:,0], X_test[:,1], c=y_hat_final,cmap=plt.cm.Accent)
plt.tight_layout()
plt.show()

