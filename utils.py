import torch                   #PyTorch base libraries
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

import cv2

from albumentations import Compose, PadIfNeeded, RandomCrop, Normalize, HorizontalFlip, ShiftScaleRotate, CoarseDropout,Cutout
from albumentations.pytorch.transforms import ToTensorV2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pl_bolts.datamodules import CIFAR10DataModule

class album_Compose_train():
    def __init__(self):
        self.albumentations_transform = Compose([
          PadIfNeeded(40),
          RandomCrop(32,32),
          HorizontalFlip(),
          Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=[0.4914*255, 0.4822*255, 0.4471*255], always_apply=True, p=0.50),
#             HorizontalFlip(),
#             ShiftScaleRotate(),
# #            Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=[0.4914*255, 0.4822*255, 0.4471*255], always_apply=True, p=1.00),
#             CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=[0.4914*255, 0.4822*255, 0.4471*255], mask_fill_value=None, always_apply=False, p=0.3),
          Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
          ToTensorV2()
        ])
    def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img

class album_Compose_test():
    def __init__(self):
        self.albumentations_transform = Compose([
            Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
            ToTensorV2()
        ])

    def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img


class dataset_cifar10:
    """
    Class to load the data and define the data loader
    """

    def __init__(self, batch_size=128):

        # Defining CUDA
        cuda = torch.cuda.is_available()
        print("CUDA availability ?",cuda)

        # Defining data loaders with setting
        self.dataloaders_args = dict(shuffle=True, batch_size = batch_size, num_workers = 2, pin_memory = True) if cuda else dict(shuffle=True,batch_size = batch_size)
        self.sample_dataloaders_args = self.dataloaders_args.copy()

        self.classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def data(self, train_flag):

        # Transformations data augmentation (only for training)
        if train_flag :
            return datasets.CIFAR10('./Data',
                            train=train_flag,
                            transform=album_Compose_train(),
                            download=True)

        # Testing transformation - normalization adder
        else:
            return datasets.CIFAR10('./Data',
                                train=train_flag,
                                transform=album_Compose_test(),
                                download=True)

    # Dataloader function
    def loader(self, train_flag=True):
        return(torch.utils.data.DataLoader(self.data(train_flag), **self.dataloaders_args))


    def data_summary_stats(self):
        # Load train data as numpy array
        train_data = self.data(train_flag=True).data
        test_data = self.data(train_flag=False).data

        total_data = np.concatenate((train_data, test_data), axis=0)
        print(total_data.shape)
        print(total_data.mean(axis=(0,1,2))/255)
        print(total_data.std(axis=(0,1,2))/255)

    def sample_pictures(self, train_flag=True, return_flag = False):

        # get some random training images
        images,labels = next(iter(self.loader(train_flag)))

        sample_size=25 if train_flag else 5

        images = images[0:sample_size]
        labels = labels[0:sample_size]

        fig = plt.figure(figsize=(10, 10))

        # Show images
        for idx in np.arange(len(labels.numpy())):
            ax = fig.add_subplot(5, 5, idx+1, xticks=[], yticks=[])
            npimg = unnormalize(images[idx])
            ax.imshow(npimg, cmap='gray')
            ax.set_title("Label={}".format(str(self.classes[labels[idx]])))

        fig.tight_layout()  
        plt.show()

        if return_flag:
            return images,labels

def unnormalize(img):
    channel_means = (0.4914, 0.4822, 0.4471)
    channel_stdevs = (0.2469, 0.2433, 0.2615)
    img = img.numpy().astype(dtype=np.float32)
  
    for i in range(img.shape[0]):
         img[i] = (img[i]*channel_stdevs[i])+channel_means[i]
  
    return np.transpose(img, (1,2,0))

# define a function to plot misclassified images
def plot_misclassified_images(model, test_loader, classes, device):
    # set model to evaluation mode
    model.eval()

    misclassified_images = []
    actual_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    misclassified_images.append(data[i])
                    actual_labels.append(classes[target[i]])
                    predicted_labels.append(classes[pred[i]])

    # Plot the misclassified images
    fig = plt.figure(figsize=(12, 5))
    for i in range(10):
        sub = fig.add_subplot(2, 5, i+1)
        npimg = unnormalize(misclassified_images[i].cpu())
        plt.imshow(npimg, cmap='gray', interpolation='none')
        sub.set_title("Actual: {}, Pred: {}".format(actual_labels[i], predicted_labels[i]),color='red')
    plt.tight_layout()
    plt.show()

def plot_grad_cam_images_custom_resnet(model, test_loader, classes, device):
    # set model to evaluation mode
    model.eval()
    target_layers = [model.R3]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)

    misclassified_images = []
    actual_labels = []
    actual_targets = []
    predicted_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    actual_targets.append(target[i])
                    misclassified_images.append(data[i])
                    actual_labels.append(classes[target[i]])
                    predicted_labels.append(classes[pred[i]])

    # Plot the misclassified images
    fig = plt.figure(figsize=(12, 5))
    for i in range(10):
        sub = fig.add_subplot(2, 5, i+1)
        input_tensor = misclassified_images[i].unsqueeze(dim=0)
        targets = [ClassifierOutputTarget(actual_targets[i])]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(unnormalize(misclassified_images[i].cpu()), grayscale_cam, use_rgb=True, image_weight=0.7)

        # npimg = unnormalize(misclassified_images[i].cpu())
        # plt.imshow(npimg, cmap='gray', interpolation='none')

        # npimg = unnormalize(misclassified_images[i].cpu())
        plt.imshow(visualization)
        sub.set_title("Actual: {}, Pred: {}".format(actual_labels[i], predicted_labels[i]),color='red')
    plt.tight_layout()
    plt.show()

def plot_grad_cam_images(model, test_loader, classes, device):
    # set model to evaluation mode
    model.eval()
    target_layers = [model.layer4[-1]]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)

    misclassified_images = []
    actual_labels = []
    actual_targets = []
    predicted_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    actual_targets.append(target[i])
                    misclassified_images.append(data[i])
                    actual_labels.append(classes[target[i]])
                    predicted_labels.append(classes[pred[i]])

    # Plot the misclassified images
    fig = plt.figure(figsize=(12, 5))
    for i in range(10):
        sub = fig.add_subplot(2, 5, i+1)
        input_tensor = misclassified_images[i].unsqueeze(dim=0)
        targets = [ClassifierOutputTarget(actual_targets[i])]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(unnormalize(misclassified_images[i].cpu()), grayscale_cam, use_rgb=True, image_weight=0.7)

        # npimg = unnormalize(misclassified_images[i].cpu())
        # plt.imshow(npimg, cmap='gray', interpolation='none')

        # npimg = unnormalize(misclassified_images[i].cpu())
        plt.imshow(visualization)
        sub.set_title("Actual: {}, Pred: {}".format(actual_labels[i], predicted_labels[i]),color='red')
    plt.tight_layout()
    plt.show()


def plot_loss_accuracy_graph(trainObj, testObj, EPOCHS):

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    train_losses = [temp.cpu().detach() for temp in trainObj.train_losses]

    train_epoch_linspace = np.linspace(1, EPOCHS, len(train_losses))
    test_epoch_linspace = np.linspace(1, EPOCHS, len(testObj.test_losses))

    # Loss Plot
    ax[0].plot(train_epoch_linspace, train_losses, label='Training Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss vs. Epochs')
    ax[0].legend()

    ax2 = ax[0].twinx()
    ax2.plot(test_epoch_linspace, testObj.test_losses, label='Test Loss', color='red')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='center right')

    # Accuracy Plot
    #train_acc = [temp.cpu().detach() for temp in trainObj.train_acc]
    ax[1].plot(train_epoch_linspace, trainObj.train_acc, label='Training Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy vs. Epochs')
    ax[1].legend()

    ax2 = ax[1].twinx()
    ax2.plot(test_epoch_linspace, testObj.test_acc, label='Test Accuracy', color='red')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='center right')

    plt.tight_layout()
    plt.show()

def plot_loss_accuracy_graph_OneCLR(trainAcc, trainLoss, testAcc, testLoss):

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Loss Plot
    ax[0].plot(trainLoss, label='Training Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss vs. Epochs')
    ax[0].legend()

    ax2 = ax[0].twinx()
    ax2.plot(testLoss, label='Test Loss', color='red')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='center right')

    # Accuracy Plot
    ax[1].plot(trainAcc, label='Training Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy vs. Epochs')
    ax[1].legend()

    ax2 = ax[1].twinx()
    ax2.plot(testAcc, label='Test Accuracy', color='red')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='center right')

    plt.tight_layout()
    plt.show()