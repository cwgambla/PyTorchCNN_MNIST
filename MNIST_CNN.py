import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms
import torch.optim as optim
import numpy as np

torch.manual_seed(42)


# train_x = torch.tensor(np.array([np.random.rand(1,28*28),np.random.rand(1,28*28)]))
# train_y = torch.tensor(np.array([[1],[0]]))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081))
])

Lambda = (lambda y: torch.zeros(10, dtype=torch.float).scatter_(0,torch.tensor(y),value=1))
# train = MNIST(root=".",train=True,transform=transform,download=True)
# test = MNIST(root=".",train=False,transform=transform,download=True)


train = MNIST(root=".",train=True,transform=transform,download=True, target_transform=Lambda)
test = MNIST(root=".",train=False,transform=transform,download=True, target_transform=Lambda)
# train = torch.tensor(train, dtype=torch.float)
# test = torch.tensor(test, dtype=torch.float)
class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10),

            #only comment out this layer if CCELoss is used
            nn.Sigmoid() 
        )


    def forward(self, x):

        #flattens input
        x = self.flatten(x)

        #runs the input throught the neural net
        #and then returns the expected output
        logits = self.layers(x)
        return logits
    

model = myModel()

loss_fn = nn.BCELoss()#Means Squared Error Function

optimizer = optim.Adam(model.parameters(), lr=0.001)#optimizer used to update weights, using Scholastic gradient decent

batch_size = 16

train_loader = DataLoader(train,batch_size=batch_size,shuffle=True)

test_loader = DataLoader(test,batch_size=batch_size)



#trainginng mooodel
epochs = 1
for epoch in range(epochs):

    #predicting x
    for index, (inputs, targets) in enumerate(train_loader):
        
        y_pred = model(inputs)
        loss = loss_fn(y_pred,targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (index + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{index+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


#testing accuracy
with torch.no_grad():
    correct = 0
    for input, target in test_loader:
        pred = model(input)
        pred = torch.argmax(pred,dim=1)
        target = torch.argmax(target,dim=1)
        # print(pred)
        # print(target)
        # print(torch.argmax(target,dim=1))
        correct += pred.eq(target.data.view_as(pred)).sum()
        

    print(correct/len(test_loader.dataset))