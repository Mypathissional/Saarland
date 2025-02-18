import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import ipdb
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
input_size = 32 * 32 * 3
layer_config = [512, 256]
num_classes = 10
num_epochs = 30
batch_size = 200
learning_rate = 1e-3
learning_rate_decay = 0.99
reg = 0  # 0.001
num_training = 49000
num_validation = 1000
fine_tune = True
pretrained = True

# , transforms.RandomGrayscale(p=0.05)]
data_aug_transforms = [transforms.RandomHorizontalFlip(p=0.5)]
# -------------------------------------------------
# Load the CIFAR-10 dataset
# -------------------------------------------------
# Q1,
norm_transform = transforms.Compose(data_aug_transforms + [transforms.ToTensor(),
                                                           transforms.Normalize([0.485, 0.456, 0.406], [
                                                                                0.229, 0.224, 0.225]),
                                                           ])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                             train=True,
                                             transform=norm_transform,
                                             download=False)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                            train=False,
                                            transform=norm_transform
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


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class VggModel(nn.Module):
    def __init__(self, n_class, fine_tune, pretrained=True):
        super(VggModel, self).__init__()
        #################################################################################
        # TODO: Build the classification network described in Q4 using the              #
        # models.vgg11_bn network from torchvision model zoo as the feature extraction  #
        # layers and two linear layers on top for classification. You can load the      #
        # pretrained ImageNet weights based on the pretrained flag. You can enable and  #
        # disable training the feature extraction layers based on the fine_tune flag.   #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        vgg11 = models.vgg11_bn(pretrained=pretrained)
        self.vgg11_bn = vgg11.features ## output of the shame [1,512,1,1]
        self.classifier = nn.Sequential(
                            nn.Linear(512, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            nn.Linear(512, n_class)
                        )

        if not fine_tune:
            for p in self.vgg11_bn.parameters():
                p.requires_grad = False


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        features = self.vgg11_bn(x)
        features = torch.reshape(features,[-1,512])
        out = self.classifier(features)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out

possible_models = [[0,1],[1,0],[1,1]]
    # Initialize the model for this run
train = []
val = []
test = []
l = ["_non_", "_"]


for i in possible_models:
    [pretrained,fine_tune] = i
    model = VggModel(num_classes, fine_tune, pretrained)

    # Print the model we just instantiated
    print(model)

    #################################################################################
    # TODO: Only select the required parameters to pass to the optimizer. No need to#
    # update parameters which should be held fixed (conv layers).                   #
    #################################################################################
    training_loss = []
    validation_accuracy = []

    if fine_tune:
        params_to_update = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        params_to_update.extend(model.parameters())
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    else:
        params_to_update = model.parameters()
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)


    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params_to_update, lr=learning_rate, weight_decay=reg)

    # Train the model
    lr = learning_rate
    total_step = len(train_loader)
    best_acc = 0
    for epoch in range(num_epochs):
        batch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
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

            if (i + 1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

            batch_loss+=loss.item()
        training_loss.append(batch_loss/batch_size)
        # Code to update the lr
        lr *= learning_rate_decay
        update_lr(optimizer, lr)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            validation_accuracy.append(100 * correct / total)
            #################################################################################
            # TODO: Q2.b Use the early stopping mechanism from previous questions to save   #
            # the model which has acheieved the best validation accuracy so-far.            #
            #################################################################################
            best_model = None
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            if best_acc < correct:
                torch.save(model.state_dict(), "model4_2"+l[pretrained]+"pretrained"
                    + "and" + l[fine_tune]+"finetuned.ckpt" )
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            print('Validataion accuracy is: {} %'.format(100 * correct / total))

    train.append(training_loss)
    val.append(validation_accuracy)

for i in range(len(possible_models)):
    label = "model"+l[pretrained]+"pretrained_and" + l[fine_tune]+"finetuned"
    print(label)
    [pretrained,fine_tune] = possible_models[i]
    plt.subplot(2, 1, 1)
    plt.plot(train[i], label=label)
    plt.title('Loss history')
    plt.xlabel('Iterations per epoch for dropot')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(val[i], label=label)
    plt.title('Classification accuracy history for dropout')
    plt.xlabel('Iterations per epoch')
    plt.legend()
plt.show()

    #################################################################################
    # TODO: Use the early stopping mechanism from previous question to load the     #
    # weights from the best model so far and perform testing with this model.       #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)

for i in range(len(possible_models)):
    [pretrained,fine_tune] = possible_models[i]
    print(pretrained, fine_tune)
    model = VggModel(num_classes, fine_tune, pretrained).cuda()
    modelName = "model4_2"+l[pretrained]+"pretrained"+ "and" + l[fine_tune]+"finetuned.ckpt"
    model.load_state_dict(torch.load(modelName))
    model.eval()

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

        print('Accuracy of the network on the {} test images of the model{}: {} %'.format(
            total,  l[pretrained]+"pretrained"+ "and" + l[fine_tune]+"finetuned",100 * correct / total))
