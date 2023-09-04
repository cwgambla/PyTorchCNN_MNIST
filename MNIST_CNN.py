import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
import numpy as np

torch.manual_seed(39)


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

class CNN(nn.Module):

    def __init__(self, in_channels = 1, num_class = 10):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(5,5),padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(5,5),padding=1)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(5,5),padding=1)
        self.conv4= nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(5,5),padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride = (2,2))
        self.fc1 = nn.Linear(1024,120)
        self.fc2 = nn.Linear(120,40)
        self.fc3 = nn.Linear(40,num_class)
        

        self.drop = nn.Dropout(.25)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)


        return x
    

model = CNN()

x = torch.randn(64,1,28,28)
print(model(x).shape)

model = CNN()

loss_fn = nn.CrossEntropyLoss()#Means Squared Error Function

optimizer = optim.Adam(model.parameters(), lr=0.001)#optimizer used to update weights, using Scholastic gradient decent

batch_size = 16

train_loader = DataLoader(train,batch_size=batch_size,shuffle=True)

test_loader = DataLoader(test,batch_size=batch_size)



#trainginng mooodel
epochs = 10
for epoch in range(epochs):

    #predicting x
    for index, (inputs, targets) in enumerate(train_loader):
        
        y_pred = model(inputs)
        loss = loss_fn(y_pred,targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (index + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{index+1}/{len(train_loader)}], Loss: {loss.item():.6f}')


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