#import
import os
import re
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
sys.path.insert(1, "/home/jupyter/ee_segmentation/models/models")
from Models import model_Selection
def dice_loss(predictions, ground_truth):
    """
    Dice loss
    
    Prameters:
    prediction: Pytorch tensor | Prediction made by a model
    groud_truth: Pytorch tensor | Ground truth
    
    Output
    dice_loss: Pytorch float | Dice Loss
    """

    eps = 1

    ground_truth = torch.squeeze(ground_truth, dim=1)

    intersection = (predictions * ground_truth).sum()
    dice_coef = (2.0 * intersection + eps) / (
        (predictions).sum() + (ground_truth).sum() + eps
    )
    dice_loss = 1 - dice_coef
    return dice_loss

def dice_loss2(predictions, ground_truth, eps=1):
    intersection = (predictions * ground_truth).sum()
    dice_coef = (2.0 * intersection + eps) / (
        predictions.sum() + ground_truth.sum() + eps)
    dice_loss = 1 - dice_coef
    return dice_loss


#eval model
@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """
    Evaluate the performance of a model for a segmentation tasks
    
    Parameters:
    model: Pytorch model | Model to evaluate
    dataloader: Pytorch data loader | Data on which we are going to perform our eveluation
    
    Outputs:
    iou: float | IOU of the predictions vs labels
    accuracy: float | accuracy of the predictions vs labels
    
    """
    intersection_total, union_total = 0, 0
    pixel_correct, pixel_count = 0, 0

    for data in dataloader:
        imgs, masks = data
        imgs, masks = imgs.to(device), masks.to(device)

        predictions = model(imgs)
        predictions = predictions > 0.5
        masks = torch.squeeze(masks, dim=0)

        intersection_total += (predictions * masks).sum()
        union_total += (predictions + masks).sum()

        pixel_correct += (predictions == masks).sum()
        pixel_count += masks.numel()

    iou = (intersection_total / union_total).item()
    accuracy = (pixel_correct / pixel_count).item()

    return iou, accuracy

def save_model_and_metrics(model, model_name, epoch, metrics, device, losses):
    folder_dir = os.path.join(os.curdir, "..", "..", "models", "trainned models")
    cur_dir = os.path.abspath(os.curdir)
    try:
        # os.chdir()
        os.chdir(os.path.join(folder_dir, model_name))
    except:
        os.mkdir(os.path.join(folder_dir, model_name))
        print(f"{model_name} folder created")
        os.chdir(os.path.join(folder_dir, model_name))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(model_name + "_metrics.csv")

    np.savetxt(f"{model_name}_losses.csv", losses)

    model.to("cpu")
    torch.save(model.state_dict(), f"{model_name}_model_epoch{epoch}.pt")
    model.to(device)

    os.chdir(cur_dir)
def save_model_and_metrics_multi_gpu(model, model_name, epoch, metrics, device, losses):
    folder_dir = os.path.join(os.curdir, "..", "..", "models", "trainned models")
    cur_dir = os.path.abspath(os.curdir)
    try:
        # os.chdir()
        os.chdir(os.path.join(folder_dir, model_name))
    except:
        os.mkdir(os.path.join(folder_dir, model_name))
        print(f"{model_name} folder created")
        os.chdir(os.path.join(folder_dir, model_name))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(model_name + "_metrics.csv")

    np.savetxt(f"{model_name}_losses.csv", losses)

    model.to("cpu")
    torch.save(model.module.state_dict(), f"{model_name}_model_epoch{epoch}.pt")
    model.to(device)

    os.chdir(cur_dir)

def initiate_dirs(model_name):
    """ "

    Parameters:
    model_name: String | name of the model

    Create the folder in which the trained models and folders to put the data into


    """

    # create dir for the model
    cur_dir = os.path.abspath(os.curdir)
    model_dir = os.path.join("..", "..", "models", "trainned models", model_name)
    try:
        os.chdir(model_dir)
        print("Folder for this model is already created")
    except:
        os.mkdir(model_dir)
        print(f"{model_name} folder created")
        os.chdir(model_dir)

    # create a data folder
    try:
        os.mkdir("data")
    except:
        pass

    os.chdir("data")
    # create the folder for train and val set for bands and masks
    data_dir = os.path.abspath(os.curdir)
    folder_list = ["bands", "masks"]
    set_list = ["train", "val", "test"]
    for folder in folder_list:
        try:
            os.mkdir(folder)
        except:
            pass
        os.chdir(folder)
        for sett in set_list:
            try:
                os.mkdir(sett)
            except:
                pass
        os.chdir(data_dir)

    os.chdir(cur_dir)

def fetch_best_model(model_name, best_epoch, model_type, Bands, backbone_network):
    """
    Parameters:
    model_name: String | Folder name in which the metrics and weights are stored 
    best_epoch: Int | Epoch with the bes val IOU
    model_type: String | Type of model used
    Bands: List | Bands took from the image
    backbone_network:String or Int | back bone network used
    
    Outputs:
    model: pytorch model | best model loaded with weights
    """
    cur_dir = os.path.abspath(os.curdir)
    folder_dir = os.path.join(
        os.curdir, "..", "..", "models", "trainned models", model_name
    )
    os.chdir(folder_dir)
    liste = os.listdir()
    r = re.compile(f".+epoch{best_epoch}.pt")
    best_model = list(filter(r.match, liste))[0]
    model = model_Selection(model_type, Bands, backbone_network, trained=False)
    model.load_state_dict(torch.load(best_model))
    os.chdir(cur_dir)
    return model
def delete_low_perf_model(model_name):
    cur_dir = os.path.abspath(os.curdir)
    model_dir = os.path.join("..", "..", "models", "trainned models", model_name)
    os.chdir(model_dir)

    r = re.compile(".+metrics\.csv")
    metrics_file = list(filter(r.match, os.listdir()))[0]
    metrics = pd.read_csv(metrics_file)
    best_epoch = metrics[metrics["Validation IOU"] == metrics["Validation IOU"].max()][
        "Epoch"
    ].iloc[0]
    file_list = os.listdir()
    for file in file_list:
        if file[-3:] == ".pt":
            if file.find("epoch" + str(best_epoch) + ".pt") > 0:
                pass
            else:
                os.remove(file)
    os.chdir(cur_dir)