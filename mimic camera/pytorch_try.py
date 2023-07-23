import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

import cv2
import pandas as pd
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import math
import pickle
#########initialization of the module##########

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(128000, 64)
        self.fc2 = nn.Linear(64, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features  # Load the pre-trained VGG-16 model's features
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(512, 64, 3)  
        self.conv4 = nn.Conv2d(64, 7, 3)  
        self.fc1 = nn.Linear(128000, 64)  
        self.fc2 = nn.Linear(64, 7)

    def forward(self, x):
        x = self.vgg(x)  # Extract features using VGG-Net model
        x = self.pool(x)
        x = self.conv3(x)  
        x = self.pool(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)  
        x = self.fc2(x)
        return x

net = Net()
'''
net = Net()

net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)


image_list = []
param_list = []
train_indices = []
temp_indices = []


'''
data selection
'''
select_images = {
    "distortion+translation": 0.1, 
    "only_translation": 0.9,
    "only_distortion": 0.9,
    "one_corner_out": 0.1,
    "two_corner_out": 0.1,
    "big_distortion_big_translation": 0.1,
    "nightmare": 0.05
}

###following are available case

#original
#3 by 5
#6 by 10/original
#6 by 10/with color distribution/1
#12 by 20
#24 by 40
#original


case_name = "6 by 10/original"
PATH = "datasets/Testset/" + case_name +'/best_model.pth'


for folder_name, fraction in select_images.items():
    print(f"{folder_name}******")
    folder_path = os.path.join("datasets/Testset/" + case_name, folder_name)
    files = os.listdir(folder_path)
    print(len(files))
    num_images = math.ceil(fraction * len(files))
    selected_files = random.sample(files, num_images)

    for i, file_name in enumerate(selected_files):
        img = cv2.imread(os.path.join(folder_path, file_name))
        img = cv2.resize(img, (360, 440))
        image_list.append(img)
        param = file_name[:-4].split()
        param = [float(p) for p in param]
        param_list.append(param)

X = np.array(image_list) 
####!!!!!!!!!!!!!!!!!!!!! need to normalized the dataset before the training, from 0-255 to 0-1
Y = np.array(param_list)
print(Y.shape)
scaler = StandardScaler()
scaler.fit(Y)
Y_scale = scaler.transform(Y)


with open('{}/{}'.format("datasets/Testset/" + case_name, "params"), 'wb') as file:
    print(scaler.mean_)
    print(scaler.var_)
    pickle.dump(scaler, file)


tensor_x = torch.Tensor(X).permute(0, 3, 1, 2) # PyTorch expects images in (N, C, H, W) format
tensor_y = torch.Tensor(Y_scale)

X_train, X_temp, Y_train, Y_temp = train_test_split(tensor_x, tensor_y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

trainset = TensorDataset(X_train, Y_train)
valset = TensorDataset(X_val, Y_val)
testset = TensorDataset(X_test, Y_test)

trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
valloader = DataLoader(valset, batch_size=16, shuffle=False)
testloader = DataLoader(testset, batch_size=16, shuffle=False)



def compute_the_loss():

    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.to(device)
    net.eval()
    running_val_losses = np.zeros(7)
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device) #double checking
            outputs = net(inputs)
            val_losses = [criterion(outputs[:, j], labels[:, j]) for j in range(7)] # Compute losses for each parameter
            for j, val_loss in enumerate(val_losses):
                running_val_losses[j] += val_loss.item()

            if (i+1) % 10 == 0:  # Print every 10 batches
                print(f'[{epoch + 1}, {i + 1}] val losses: {running_val_losses / 10}')
                with open("datasets/Testset/" + case_name + '/batch_losses.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    row = [epoch] + list(running_val_losses / 10)
                    writer.writerow(row)
                running_val_losses.fill(0.0)

######## Set up the training parameters ###########

train_losses = []
val_losses = []

with open("datasets/Testset/" + case_name + '/batch_losses.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    header = ['type', 'epoch'] + [f'loss_{i+1}' for i in range(7)]
    writer.writerow(header)

n_epochs_stop = 100
epochs_no_improve = 0
best_val_loss = float('inf')
early_stop = False
best_model = None

total_epochs = 100
checkpoints = 5
epochs_per_checkpoint = total_epochs // checkpoints


for epoch in range(total_epochs): 
    running_losses = np.zeros(7)
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)

        losses = [criterion(outputs[:, j], labels[:, j]) for j in range(7)] 
        combined_loss = sum(losses)
        combined_loss.backward() 

        optimizer.step()

        for j, loss in enumerate(losses):
            running_losses[j] += loss.item()
        

        with open("datasets/Testset/" + case_name + '/batch_losses.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            row = ['train', epoch] + list(running_losses / ((i % 10) + 1))
            writer.writerow(row)

        if (i+1) % 10 == 0:
            print(f'[{epoch + 1}, {i + 1}] train losses: {running_losses / 10}')
            running_losses.fill(0.0)

    net.eval()
    running_val_losses = np.zeros(7)
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            val_losses = [criterion(outputs[:, j], labels[:, j]) for j in range(7)] 
            for j, val_loss in enumerate(val_losses):
                running_val_losses[j] += val_loss.item()

        avg_val_loss = np.mean(running_val_losses)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = deepcopy(net.state_dict()) 
            torch.save(best_model, "datasets/Testset/" + case_name +'/best_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                compute_the_loss()
                early_stop = True
                break

    if epoch % epochs_per_checkpoint == 0:
        torch.save(net.state_dict(), f"datasets/Testset/{case_name}/checkpoint_epoch_{epoch}.pth") 

    with open("datasets/Testset/" + case_name + '/batch_losses.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        row = ['val', epoch] + list(running_val_losses / ((i % 10) + 1))
        writer.writerow(row)

    if (i+1) % 10 == 0:  # Print every 10 batches
        print(f'[{epoch + 1}, {i + 1}] val losses: {running_val_losses / 10}')
        running_val_losses.fill(0.0)

    if early_stop:
        print("Stopped training due to early stopping.")
        break

print('#########Finished Training#############')

# Load the best model
net.load_state_dict(torch.load("datasets/Testset/" + case_name + '/best_model.pth'))
print('Loaded the best model.')


######draw the graph and calculate the loss#########
df = pd.read_csv("datasets/Testset/" + case_name + '/batch_losses.csv')

# Separate training and validation
df_train = df[df['type'] == 'train']
df_val = df[df['type'] == 'val']

plt.figure(figsize=(12, 6))
for param in range(1, 8):
    plt.plot(df_train[f'loss_{param}'], label=f'Training loss - Parameter {param}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses for Each Parameter')
plt.legend()
plt.show()

# Parameter in a separate subplot
fig, axs = plt.subplots(7, 1, figsize=(12, 6 * 7))
for param in range(1, 8):
    ax = axs[param - 1]
    ax.plot(df_train[f'loss_{param}'], label='Training loss')
    ax.plot(df_val[f'loss_{param}'], label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Losses for Parameter {param}')
    ax.legend()
plt.tight_layout()
plt.show()

compute_the_loss()



