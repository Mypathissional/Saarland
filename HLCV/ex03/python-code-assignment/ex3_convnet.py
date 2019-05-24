import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import ipdb
from tqdm import tqdm
from torchsummary import summary


def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# --------------------------------
# Device configuration
# --------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

# --------------------------------
# Hyper-parameters
# --------------------------------
input_size = 3
num_classes = 10
hidden_size = [128, 512, 512, 512, 512]
num_epochs = 20
batch_size = 200
learning_rate = 2e-3
learning_rate_decay = 0.95
reg = 0.001
num_training = 49000
num_validation = 1
#num_validation = 1000
norm_layer = None
print(hidden_size)
data_aug_enabled = False

# -------------------------------------------------
# Load the CIFAR-10 dataset
# -------------------------------------------------
#################################################################################
# TODO: Q3.a Chose the right data augmentation transforms with the right        #
# hyper-parameters and put them in the data_aug_transforms variable             #
#################################################################################
data_aug_transforms = []
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
if data_aug_enabled:
    data_aug_transforms = data_aug_transforms + [
        transforms.ColorJitter(),
        transforms.RandomAffine(-180, translate=(-0.1, 0.1), scale=(-0.1, 0.1))
    ]

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
norm_transform = transforms.Compose(data_aug_transforms + [transforms.ToTensor(),
                                                           transforms.Normalize(
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                             train=True,
                                             transform=norm_transform,
                                             download=False)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                            train=False,
                                            transform=test_transform
                                            )
# -------------------------------------------------
# Prepare the training and validation splits
# -------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

# -------------------------------------------------
# Data loader
# -------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# -------------------------------------------------
# Convolutional neural network (Q1.a and Q2.a)
# Set norm_layer for different networks whether using batch normalization
# -------------------------------------------------

class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, norm_layer=None,
                 use_dropout=False, p=0.1):
        super(ConvNet, self).__init__()
        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module.                          #
        #################################################################################

        layers = []

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dims = [input_size]
        dims.extend(hidden_layers)
        self.input_size = input_size

        for i in range(len(dims) - 1):
            [hid_in, hid_out] = [dims[i], dims[i + 1]]
            layers.append(self.__convBlock(hid_in, hid_out, norm_layer))
            if use_dropout:
                layers.append(nn.Dropout(p))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_layers[-1], num_classes)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def __convBlock(self, in_channel, out_channel, norm_layer, kernel_size=3, stride=1, padding=1):
        if norm_layer:
            seq = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
                                nn.BatchNorm2d(out_channel),
                                nn.MaxPool2d(kernel_size=(2, 2),
                                             stride=(2, 2)),
                                nn.ReLU())
        else:
            seq = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
                                nn.MaxPool2d(kernel_size=(2, 2),
                                             stride=(2, 2)),
                                nn.ReLU())
        return seq

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out = self.features(x)
        out = out.view(x.size(0), -1)
        out = self.classifier(out)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out


# -------------------------------------------------
# Calculate the model size (Q1.b)
# if disp is true, print the model parameters, otherwise, only return the number of parameters.
# -------------------------------------------------
def PrintModelSize(model):
    #################################################################################
    # TODO: Implement the function to count the number of trainable parameters in   #
    # the input model. This useful to track the capacity of the model you are       #
    # training                                                                      #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    model_sz = 0
    for i, param in enumerate(model.parameters()):
        if param.requires_grad:
            model_sz += param.numel()
    print("Total number of params: ", model_sz)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return model_sz

# -------------------------------------------------
# Calculate the model size (Q1.c)
# visualize the convolution filters of the first convolution layer of the input model
# -------------------------------------------------

def VisualizeFilter(model, vert=16.):
    #################################################################################
    # TODO: Implement the functiont to visualize the weights in the first conv layer#
    # in the model. Visualize them as a single image fo stacked filters.            #
    # You can use matlplotlib.imshow to visualize an image in python                #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    first_conv = next(model.parameters())
    first_conv = first_conv.detach().cpu().numpy().transpose(0, 3, 1, 2)
    interp = lambda a: np.interp(a, (a.min(), a.max()), (0., 1.))
    first_conv = np.apply_along_axis(interp,1,first_conv)

    fig, axs = plt.subplots(int(ceil(first_conv.shape[0] / vert)), int(vert))
    for i in range(int(first_conv.shape[0] / vert)):
        for j in range(int(vert)):
            axs[i][j].imshow(first_conv[i * int(vert) + j])
    plt.show()

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # ======================================================================================
    # Q1.a: Implementing convolutional neural net in PyTorch
    # ======================================================================================
    # In this question we will implement a convolutional neural networks using the PyTorch
    # library.  Please complete the code for the ConvNet class evaluating the model
    # --------------------------------------------------------------------------------------
model = ConvNet(input_size, hidden_size, num_classes,
                norm_layer=norm_layer).to(device)
# Q2.a - Initialize the model with correct batch norm layer

model.apply(weights_init)
# Print the model
print(model)
# Print model size
# ======================================================================================
# Q1.b: Implementing the function to count the number of trainable parameters in the model
# ======================================================================================
PrintModelSize(model)
# ======================================================================================
# Q1.a: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
# ======================================================================================
#VisualizeFilter(model)
summary(model, (3,32,32))


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=reg)

# Train the model
lr = learning_rate
total_step = len(train_loader)
best_acc = 0
training_loss = []
validation_accuracy = []
for epoch in range(num_epochs):
    batch_loss = 0
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        # Move tensors to the configured device

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()
        if (i + 1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    training_loss.append(batch_loss/batch_size)
    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        validation_accuracy.append(100 * correct / total)
        print('Validataion accuracy is: {} %'.format(100 * correct / total))
        #################################################################################
        # TODO: Q2.b Implement the early stopping mechanism to save the model which has #
        # acheieved the best validation accuracy so-far.                                #
        #################################################################################
        best_model = None
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        if best_acc < correct:
            torch.save(model.state_dict(), 'model1_1.ckpt')
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    model.train()

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)

model.load_state_dict(torch.load('model1_1.ckpt'))
model.eval()
#################################################################################
# TODO: Q2.b Implement the early stopping mechanism to load the weights from the#
# best model so far and perform testing with this model.                        #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
            break

    print('Accuracy of the network on the {} test images: {} %'.format(
        total, 100 * correct / total))

# Q1.c: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
plt.subplot(2, 1, 1)
plt.plot(training_loss)
plt.title('Loss history')
plt.xlabel('Iterations per epoch')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(validation_accuracy, label='val')
plt.title('Classification accuracy history for params')
plt.xlabel('Iterations per epoch')
plt.show()

VisualizeFilter(model)
# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')
