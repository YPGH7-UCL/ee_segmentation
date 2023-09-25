import os
import re
import torch
import torchgeo
import torchvision
from Seg_Net import SegNet
from Far_seg import FarSeg
from ultralytics import YOLO
from torchgeo.models import ResNet18_Weights

def model_Selection(model_name, bands, backbone_network=-1, trained=False):
    """
    Select the model we want to use based on it name and the number of input channels

    Argument:
    model_name: String | model name
    bands: List | Bands used for the model
    backbone_network: String | backbone network and version of the nodel
    trained: Bool | load weights or not

    Return:
    model: model | ML model
    """
    Models = {
        "text": ["U-Net", "Seg-Net", "FarSeg", "YOLO V5", "YOLO V8"],
        "function": [UNet, Seg__Net, FarSeg_local, YoloV5, YoloV8],
    }
    in_Channels = len(bands)
    out_Channels = 1
    assert (
        Models["text"].count(model_name) > 0
    ), f"{model_name} is not a model available"
    model = Models["function"][Models["text"].index(model_name)](
        in_Channels, out_Channels, backbone_network
    )
    print(model_name)
    if trained == True and model_name != "Seg-Net":
        curdir = os.path.abspath(os.curdir)
        weights_dir = os.path.join("..", "..", "models", "models", model_name + "_weights")
        os.chdir(weights_dir)
        liste = os.listdir()
        if backbone_network != -1:
            r = re.compile(".*" + backbone_network)
            weights = list(filter(r.match, liste))[0]
        else:
            weights = liste[0]
        model.load_state_dict(torch.load(weights))
        os.chdir(curdir)
        print('Pretrained weights loaded')
    return model

def UNet(channels_in, channels_out, backbone_network):
    """
    Load a Unet model
    source:https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/

    Argument:
    channels_in: Int | number of input channels
    channels_out: List | number of output channels

    Return:
    model: model | U-Net model
    """
    if channels_in == 3:
        model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=channels_in,
            out_channels=channels_out,
            init_features=32,
            pretrained=False,
        )
    else:
        model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=channels_in,
            out_channels=channels_out,
            init_features=32,
            pretrained=False,
        )
    return model
def YoloV5(channels_in, channels_out, yolo_model="yolov5s"):
    """ " """
    yolo_models = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]

    assert yolo_model in yolo_models
    if channels_in == 3:
        model = torch.hub.load(
            "ultralytics/yolov5",
            yolo_model,
            pretrained=False,
            channels=channels_in,
            classes=channels_out,
        )
    else:
        model = torch.hub.load(
            "ultralytics/yolov5",
            yolo_model,
            pretrained=False,
            channels=channels_in,
            classes=channels_out,
        )
    return model
def YoloV8(backbone_network, channels_in, channels_out):
    if channels_in == 3:
        # model = torch.load("deeplabv3_mobilenet_v3_large-fc3c493d.pth")
        model = YOLO("yolov8n-seg.yaml")
    else:
        model = YOLO("yolov8n-seg.yaml")
    return model


def Seg__Net(channels_in, channels_out, backbone_network):
    """
    Load a Seg-net model
    source:https://github.com/vinceecws/SegNet_PyTorch/blob/master/Pavements/SegNet.py

    Argument:
    channels_in: Int | number of input channels
    channels_out: List | number of output channels

    Return:
    model: model | SegNet model
    """
    if channels_in == 3:
        model = SegNet(in_chn=channels_in, out_chn=channels_out)
    else:
        model = model = SegNet(in_chn=channels_in, out_chn=channels_out)
    return model
def FarSeg_torch(channels_in, channels_out, backbone="resnet101"):
    """
    Load a FarSeg model
    source: https://github.com/Z-Zheng/FarSeg

    Argument:
    channels_in: Int | number of input channels
    channels_out: List | number of output channels

    Return:
    model: model | FarSeg model
    """
    if backbone == -1:
        backbone = "resnet101"

    model = torchgeo.models.FarSeg(
        backbone=backbone, classes=channels_out, backbone_pretrained=False
    )
    return model
def FarSeg_local(channels_in, channels_out, backbone="resnet101"):
    """
    Load a FarSeg model
    source: https://github.com/Z-Zheng/FarSeg

    Argument:
    channels_in: Int | number of input channels
    channels_out: List | number of output channels

    Return:
    model: model | FarSeg model
    """
    if backbone == -1:
        backbone = "resnet101"

    model = FarSeg(backbone=backbone, classes=channels_out, backbone_pretrained=False)
    return model