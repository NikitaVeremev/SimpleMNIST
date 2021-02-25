import torch
import torchvision
import numpy as np

from typing import Tuple, List, Type, Dict, Any
from torchvision import datasets, models, transforms
from ConvNet import ConvNet

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_transforms = torchvision.transforms.Compose([
    transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5)),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(size=28, scale=(0.5, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

val_transforms = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

train_dataset = torchvision.datasets.MNIST(root='./mnist',
                                           train=True,
                                           download=True,
                                           transform=train_transforms)

val_dataset = torchvision.datasets.MNIST(root='./mnist',
                                         train=False,
                                         download=True,
                                         transform=val_transforms)


def train_single_epoch(model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       loss_function1: torch.nn.Module,
                       loss_function2: torch.nn.Module,
                       train_loader: torch.utils.data.DataLoader):
    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels1 = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs1 = outputs[0]
        outputs2 = outputs[1]
        labels2 = labels1.unsqueeze(1).type_as(outputs2) % 2
        loss1 = loss_function1(outputs1, labels1)
        loss2 = loss_function2(outputs2, labels2)
        loss = loss1 * 10 + loss2
        loss.backward()
        optimizer.step()


def validate_single_epoch(model: torch.nn.Module,
                          loss_function1: torch.nn.Module,
                          loss_function2: torch.nn.Module,
                          val_loader: torch.utils.data.DataLoader):
    model.eval()
    running_loss1 = 0.0
    running_corrects1 = 0
    running_loss2 = 0.0
    running_corrects2 = 0
    processed_size = 0
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        labels1 = labels.to(DEVICE)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs1 = outputs[0]
            outputs2 = outputs[1]
            labels2 = labels1.unsqueeze(1).type_as(outputs2) % 2
            loss1 = loss_function1(outputs1, labels1)
            loss2 = loss_function2(outputs2, labels2)
            preds1 = torch.argmax(outputs1, 1)
            preds2 = outputs2 > 0.5
        running_loss1 += loss1.item() * inputs.size(0)
        running_loss2 += loss2.item() * inputs.size(0)
        running_corrects1 += torch.sum(preds1 == labels1.data)
        running_corrects2 += torch.sum(preds2 == labels2.data)
        processed_size += inputs.size(0)
    val_loss1 = running_loss1 / processed_size
    val_acc1 = running_corrects1.double() / processed_size
    val_loss2 = running_loss2 / processed_size
    val_acc2 = running_corrects2.double() / processed_size
    return {"loss1": val_loss1, "accuracy1": val_acc1, "loss2": val_loss2,
            "accuracy2": val_acc2, "mean_loss": (val_loss1 + val_loss2) / 2}


def train_model(model: torch.nn.Module,
                train_dataset: torch.utils.data.Dataset,
                val_dataset: torch.utils.data.Dataset,
                loss_function1: torch.nn.Module,
                loss_function2: torch.nn.Module,
                optimizer_class: Type[torch.optim.Optimizer] = torch.optim,
                optimizer_params: Dict = {"momentum": 0.975},
                initial_lr=1e-3,
                lr_scheduler_class: Any = torch.optim.lr_scheduler.StepLR,
                lr_scheduler_params: Dict = {"step_size": 5, "gamma": 0.1},
                batch_size=64,
                max_epochs=10,
                early_stopping_patience=20):
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, **optimizer_params)
    lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_params)

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    best_val_loss = None
    best_epoch = None

    for epoch in range(max_epochs):

        print(f'Epoch {epoch}')
        train_single_epoch(model, optimizer, loss_function1, loss_function2, train_loader)
        val_metrics = validate_single_epoch(model, loss_function1, loss_function2, val_loader)
        print(f'Validation metrics: \n{val_metrics}')

        lr_scheduler.step()

        if best_val_loss is None or best_val_loss > val_metrics['mean_loss']:
            print(f'Best model yet, saving')
            best_val_loss = val_metrics['mean_loss']
            best_epoch = epoch
            torch.save(model, './best_model.pth')

        if epoch - best_epoch > early_stopping_patience:
            print('Early stopping triggered')
            return


if __name__ == "__main__":
    model = ConvNet()
    # Используем GPU
    # model = model.cuda()
    # работаем на видеокарте
    DEVICE = torch.device("cuda")

    train_model(model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                loss_function1=torch.nn.CrossEntropyLoss(),
                loss_function2=torch.nn.BCELoss(),
                initial_lr=1e-3)
