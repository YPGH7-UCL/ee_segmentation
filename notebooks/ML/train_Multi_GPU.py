from Data_Set_and_Loader import get_data, get_data2, delet_data
from utilss import (
    save_model_and_metrics,
    save_model_and_metrics_multi_gpu,
    dice_loss,
    dice_loss2,
    evaluate_model,
    initiate_dirs,
    fetch_best_model,
    delete_low_perf_model,
)

# origin = "/home/jupyter/ee_segmentation/notebooks/ML"
import sys
import os
import re
import numpy as np
import pandas as pd

sys.path.insert(1, "/home/jupyter/ee_segmentation/models/models")
from Models import model_Selection
import torch
import torch.nn as nn
from torch.optim import Adam

# def train (Bands, BatchSize, epochs, img_size, lr, w_d, model_composition):
def train(parameters):
    torch.manual_seed(115)
    np.random.seed(115)
    Bands, BatchSize, epochs, img_size, lr, w_d, model_params = parameters.get('Bands'), parameters.get('BatchSize'), parameters.get('epochs'), parameters.get('img_size'), parameters.get('lr'), parameters.get('w_d'), parameters.get('model_params')
    model_type, backbone = model_params.split('_')
    try:
        backbone = eval(backbone)
    except:
        pass
    Bands = eval(Bands)
    # Use GPU if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model_Selection(model_type, Bands, backbone_network=backbone)
    if torch.cuda.is_available():
        model.cuda()
        model = nn.DataParallel(model)
        model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=w_d)
    model_name = f"{model_type}_Bands:{Bands}_backbone:{backbone}_Image_size:{img_size}"
    # setup the folders
    initiate_dirs(model_name)
    trainloader, valloader, testloader = get_data2(
    model_name=model_name,
    bands_selection=Bands,
    batch_size=BatchSize,
    img_size=img_size,
    )
    trainloader.dataset.eval(), valloader.dataset.eval(), testloader.dataset.eval()
    EvalFraquency = 5
    Epochs, IOUs_train, accuracies_train, IOUs_val, accuracies_val, loss_rec = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    model.train()

    for epoch in range(epochs):
        running_loss = 0
        trainloader.dataset.train()
        for step, data in enumerate(trainloader):  # replace by trainloader
            imgs, masks = data
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            # make prediction

            predict = model(imgs)
            # print(predict.shape,predict.min(), predict.max())
            # calculate the loss
            loss = dice_loss2(predict, masks)
            # backpropagation

            # print(f'Loss is {loss}.')
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1} -Loss: {running_loss:.2f}")
        loss_rec.append(running_loss)
        if (epoch) % EvalFraquency == 0 or epoch == epochs - 1:
            model.eval(), trainloader.dataset.eval()
            IOU_train, accuracy_train = evaluate_model(
                model, trainloader, device
            )  # replace by trainloader
            IOU_val, accuracy_val = evaluate_model(
                model, valloader, device
            )  # replace by valloader

            model.train()
            print(
                f"For trainning dataset IOU: {IOU_train:.2%} accuracy :{accuracy_train:.2%}"
            )
            print(f"For validation dataset IOU: {IOU_val:.2%} accuracy :{accuracy_val:.2%}")

            IOUs_train.append(IOU_train)
            accuracies_train.append(accuracy_train)
            IOUs_val.append(IOU_val)
            accuracies_val.append(accuracy_val)
            Epochs.append(epoch)
            metrics = {
                "Epoch": Epochs,
                "Train IOU": IOUs_train,
                "Train Accuracy": accuracies_train,
                "Validation IOU": IOUs_val,
                "Validation Accuracy": accuracies_val,
            }
            # save IOU and accuracy
            save_model_and_metrics(model, model_name, epoch, metrics, device, loss_rec)
    delet_data(model_name)
            
    metrics = pd.DataFrame(metrics)
    best_epoch = metrics[metrics["Validation IOU"] == metrics["Validation IOU"].max()]["Epoch"].item()
    best_IOU = metrics[metrics["Validation IOU"] == metrics["Validation IOU"].max()]["Validation IOU"].item()
    best_model = fetch_best_model(model_name, best_epoch, model_type, Bands, backbone)
    delete_low_perf_model(model_name)
    return best_IOU


def train_3_bands(parameters):
    torch.manual_seed(115)
    np.random.seed(115)
    Band1, Band2, Band3, BatchSize, epochs, img_size, lr, w_d, model_params = parameters.get('Band1'), parameters.get('Band2'), parameters.get('Band3'), parameters.get('BatchSize'), parameters.get('epochs'), parameters.get('img_size'), parameters.get('lr'), parameters.get('w_d'), parameters.get('model')
    model_type, backbone = model_params.split('_')
    try:
        backbone = eval(backbone)
    except:
        pass
    Bands = [Band1, Band2, Band3]
    if img_size == 512:
        if model_type == 'U-Net':
            BatchSize = 60
        elif model_type == 'Seg-Net':
            BatchSize = 28
        elif model_type == 'FarSeg':
            if backbone == 'resnet101':
                BatchSize = 52
            else:
                BatchSize = 72
    if img_size == 1024:
        if model_type == 'U-Net':
            BatchSize = 16
        elif model_type == 'Seg-Net':
            BatchSize = 28#TBC
        elif model_type == 'FarSeg':
            if backbone == 'resnet101':
                BatchSize = 52#TBC
            else:
                BatchSize = 16
    # Use GPU if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model_Selection(model_type, Bands, backbone_network=backbone,trained=False)
    if torch.cuda.is_available():
        model.cuda()
        model = nn.DataParallel(model)
        model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=w_d)
    model_name = f"{model_type}_Bands:{Bands}_backbone:{backbone}_Image_size:{img_size}"
    # setup the folders
    initiate_dirs(model_name)
    trainloader, valloader, testloader = get_data2(
    model_name=model_name,
    bands_selection=Bands,
    batch_size=BatchSize,
    img_size=img_size,
    )
    trainloader.dataset.eval(), valloader.dataset.eval(), testloader.dataset.eval()
    EvalFraquency = 5
    Epochs, IOUs_train, accuracies_train, IOUs_val, accuracies_val, loss_rec = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    model.train()

    for epoch in range(epochs):
        running_loss = 0
        trainloader.dataset.train()
        for step, data in enumerate(trainloader):  # replace by trainloader
            imgs, masks = data
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            # make prediction

            predict = model(imgs)
            # print(predict.shape,predict.min(), predict.max())
            # calculate the loss
            loss = dice_loss2(predict, masks)
            # backpropagation

            # print(f'Loss is {loss}.')
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1} -Loss: {running_loss:.2f}")
        loss_rec.append(running_loss)
        if (epoch) % EvalFraquency == 0 or epoch == epochs - 1:
            model.eval(), trainloader.dataset.eval()
            IOU_train, accuracy_train = evaluate_model(
                model, trainloader, device
            )  # replace by trainloader
            IOU_val, accuracy_val = evaluate_model(
                model, valloader, device
            )  # replace by valloader

            model.train()
            print(
                f"For trainning dataset IOU: {IOU_train:.2%} accuracy :{accuracy_train:.2%}"
            )
            print(f"For validation dataset IOU: {IOU_val:.2%} accuracy :{accuracy_val:.2%}")

            IOUs_train.append(IOU_train)
            accuracies_train.append(accuracy_train)
            IOUs_val.append(IOU_val)
            accuracies_val.append(accuracy_val)
            Epochs.append(epoch)
            metrics = {
                "Epoch": Epochs,
                "Train IOU": IOUs_train,
                "Train Accuracy": accuracies_train,
                "Validation IOU": IOUs_val,
                "Validation Accuracy": accuracies_val,
            }
            # save IOU and accuracy
            save_model_and_metrics_multi_gpu(model, model_name, epoch, metrics, device, loss_rec)
    delet_data(model_name)
            
    metrics = pd.DataFrame(metrics)
    best_epoch = metrics[metrics["Validation IOU"] == metrics["Validation IOU"].max()]["Epoch"].item()
    best_IOU = metrics[metrics["Validation IOU"] == metrics["Validation IOU"].max()]["Validation IOU"].item()
    best_model = fetch_best_model(model_name, best_epoch, model_type, Bands, backbone)
    delete_low_perf_model(model_name)
    return best_IOU
