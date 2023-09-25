import re
import io
import os
import torch
import torchvision
import numpy as np
from google.cloud import storage
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
import matplotlib.pyplot as plt

def open_npy(typ, name, Set):
    """
    Open a .npy files

    Parameters:
    typ: String | type of matrix band or mask
    name: String | name of the file
    Set: String | from which set are we pooling the data.


    Returns:
    matrix: np array | The desired matrix
    """
    matrix = np.load(
        os.path.realpath(
            os.path.join(
                os.path.curdir,
                "..",
                "..",
                "data",
                "processed",
                typ,
                Set,
                name,
                )
            )
        )
    return matrix

def size(liste, Set):
    """
    Calculate the average size of the arrays in the mask folder and test set

    Parameters:
    liste: List | list of the mask files to analyze
    Set: String | set to analyse

    Returns:
    average: int | average size of the matrix

    """
    size = 0
    for mask in liste:
        mask = open_npy("masks", mask, Set)
        size += mask.shape[0]

    average = size / len(liste)
    return int(average)

def bands_mean_std(bands):
    """
    Calculate the mean and standard deviation of an array with several bands

    Parameters:
    bands: numpy array | Array made of data from several bands

    Returns:
    results: numpy array | Array with the average and std of the bands

    """

    bands_liste = list(bands.dtype.fields.keys())
    results = np.zeros((2), dtype=bands.dtype)
    for band in bands_liste:
        mean = np.mean(bands[band])
        std = np.std(bands[band])

        results[band] = (mean, std)

    return results

def mean_std(liste, Set, bands):
    """
    Calculate the mean and standard deviation of the arrays in the mask band and test set

    Parameters:
    liste: List | list of the mask files to analyze
    Set: String | set to analyse

    Returns:
    average: int | average size of the matrix

    """
    if os.listdir().count("Mean_std_" + Set + ".npy") > 0:
        sums = np.load("Mean_std_" + Set + ".npy")
    else:
        print(liste)
        matrix_type = open_npy("bands", liste[0], Set).dtype
        sums = np.zeros((2), dtype=matrix_type)
        bands_list = list(matrix_type.fields.keys())
        for bands_name in liste:
            band = open_npy("bands", bands_name, Set)
            meanstd = bands_mean_std(band)
            for b in bands_list:
                sums[b] += meanstd[b] / len(liste)
        np.save("Mean_std_" + Set + ".npy", sums)
    return sums

class SegmentationDataSet(Dataset):

    """ "
    Construct a dataset of images and masks

    Parameters:
    band_dir: String | Directory of the bands
    mask_dir: String | Directory of the masks
    bands: List | liste of the desired bands
    Set: String | from which set are we pooling the data
    img_size: int | output size. If < 0 average of the set
    nomalization: bool | Is normalization is required
    
    Outputs:
    img: Torch tensor | tensor of the image with the required bands
    mask: Torch tensor | tensor of the mask assosiated with the image
    """

    def __init__(
        self,
        band_dir,
        mask_dir,
        bands=["B2", "B3", "B4"],
        Set="train",
        img_size=-1,
        normalization = True,
    ):
        self.band_dir = os.path.join(band_dir, Set)
        self.mask_dir = os.path.join(mask_dir, Set)
        self.band = bands
        self.set = Set
        self.norm = normalization
        pattern = "\S+.npy"
        p = re.compile(pattern)
        self.bands = sorted([l for l in os.listdir(self.band_dir) if p.match(l)])
        self.masks = sorted([l for l in os.listdir(self.mask_dir) if p.match(l)])
        if img_size < 0:
            self.size = size(self.masks, Set)
        else:
            self.size = img_size
        self.mean_std = mean_std(self.bands, Set, bands)

    def __len__(self):
        return len(self.bands)

    def __getitem__(self, idx):
        band_name = self.bands[idx]
        mask_name = self.masks[idx]
        band_path = os.path.join(self.band_dir, band_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        bandss = open_npy("bands", band_name, self.set)
        mask = torch.from_numpy(open_npy("masks", mask_name, self.set))
        arrays_norm = [bandss[b] for b in self.band]
        Normilize = torchvision.transforms.Normalize(
            [self.mean_std[b][0] for b in self.band],
            [self.mean_std[b][1] for b in self.band],
        )
        img = np.stack(arrays_norm, axis=2)
        img = torch.from_numpy(img).permute(2, 0, 1)
        # print(img[0].mean(), img[0].std())
        if self.norm:
            img_norm = Normilize(img)
        else:
            img_norm = img
        # print(img_norm[0].mean(), img_norm[0].std())
        resize = Resize(self.size)
        img = resize(img_norm)
        mask = resize(torch.unsqueeze(mask, 0))
        return img, mask

def get_data(bands_selection, batch_size, img_size=-1, normalization=True):
    """
    Provide the user with the train, validation and test data loaders

    Parameters:
    bands_selection : List | Liste of the selected bands
    batch_size: int | number of images and masks in a batch
    img_size: int | output size. If < 0 average of the set
    normalization: bool | Is normalization is required

    Output:
    trainloader : DataLoader | DataLoader with training bands and masks
    valloader : DataLoader | Dataloader with validation bands and masks
    testloader : DataLoader | Dataloader with test bands and masks

    """
    band_dir = os.path.realpath(
        os.path.join(
            os.path.curdir,
            "..",
            "..",
            "data",
            "processed",
            "bands",
        )
    )
    mask_dir = os.path.realpath(
        os.path.join(os.path.curdir, "..", "..", "data", "processed", "masks")
    )

    train_data_set = SegmentationDataSet(
        band_dir,
        mask_dir,
        bands=bands_selection,
        Set="train",
        img_size=img_size,
        normalization=normalization,
    )
    val_data_set = SegmentationDataSet(
        band_dir,
        mask_dir,
        bands=bands_selection,
        Set="val",
        img_size=img_size,
        normalization=normalization,
    )
    test_data_set = SegmentationDataSet(
        band_dir,
        mask_dir,
        bands=bands_selection,
        Set="test",
        img_size=img_size,
        normalization=normalization,
    )
    trainloader = DataLoader(train_data_set, batch_size=batch_size)
    valloader = DataLoader(val_data_set, batch_size=batch_size)
    testloader = DataLoader(test_data_set, batch_size=batch_size)
    return trainloader, valloader, testloader

def Reduce_data_set(model_name, Bands, img_size, set_name):
    """"
    Download the desired bands and reshape the bands and masks to the desired size
    
    Parameters:
    model_name: String | Name of the model
    Bands: Liste | Liste of bands to extract
    img_size: Int | The desired image size
    set_name: String | Set name
    """
    bucket_name = "wind-turbine-project-thibaud"

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    # loop around the data folder
    cur_dir = cur_dir = os.path.abspath(os.curdir)
    model_path = os.path.join(
        "..", "..", "models", "trainned models", model_name, "data"
    )
    pattern = "\/"
    pattern2 = "\w+_\S+"
    resize = Resize([img_size, img_size])
    for dirpath, dirs, filenames in os.walk(model_path):
        if dirs == []:
            f = re.split(pattern, dirpath)
            folder, sett = f[-2], f[-1]
            # open the file from the bucket
            my_prefix = f"image & ground truth/processed/{folder}/{sett}/"
            blobs = bucket.list_blobs(prefix=my_prefix, delimiter="/")
            for blob in blobs:
                file_name = re.findall(pattern2, blob.name)
                if file_name != [] and sett == set_name:
                    blob_bytes = blob.download_as_bytes()
                    blob_stream = io.BytesIO(blob_bytes)
                    arr = np.load(blob_stream)
                    if folder == "bands":
                        arr_norm = [arr[b] for b in Bands]
                        img = np.stack(arr_norm, axis=2)
                        img = torch.from_numpy(img).permute(2, 0, 1)
                    else:
                        img = torch.from_numpy(arr)
                        img = torch.unsqueeze(img, 0)
                    img = resize(img)
                    path = dirpath + "/" + file_name[0][:-3] + "pt"
                    # print(path)
                    torch.save(img, path)

    os.chdir(cur_dir)
    
def delet_data(model_name):
    
    """
    Delete the data in a model folder
    
    Parameters:
    model_name: String | Name of the model
    
    
    """
    # loop around the data folder
    cur_dir = cur_dir = os.path.abspath(os.curdir)
    model_path = os.path.join(
        "..", "..", "models", "trainned models", model_name, "data"
    )
    pattern = "\/"
    pattern2 = "\w+_\S+"
    for dirpath, dirs, filenames in os.walk(model_path, topdown=False):
        if filenames != []:
            for file_name in filenames:
                path = dirpath + "/" + file_name
                os.remove(path)
        os.rmdir(dirpath)
    
class SegmentationDataSet2(Dataset):

    """ "
    Construct a dataset of images and masks

    Parameters:
    band_dir: String | Directory of the bands
    mask_dir: String | Directory of the masks
    bands: List | liste of the desired bands
    Set: String | from which set are we pooling the data
    img_size: int | output size. If < 0 average of the set
    nomalization: bool | Is normalization is required

    """

    def __init__(
        self,
        model_name,
        bands=["B2", "B3", "B4"],
        Set="train",
        img_size=-1,
        normalization=True,
        augmentation=True,
    ):
        self.band_dir = os.path.join(
            "..", "..", "models", "trainned models", model_name, "data", "bands", Set
        )
        self.mask_dir = os.path.join(
            "..", "..", "models", "trainned models", model_name, "data", "masks", Set
        )
        if os.listdir(self.band_dir) == []:
            Reduce_data_set(model_name, bands, img_size, Set)
        self.band = bands
        self.set = Set
        self.norm = normalization
        self.aug = augmentation
        pattern = "\S+.pt"
        p = re.compile(pattern)
        self.bands = sorted([l for l in os.listdir(self.band_dir) if p.match(l)])
        self.masks = sorted([l for l in os.listdir(self.mask_dir) if p.match(l)])
        if img_size < 0:
            self.size = size(self.masks, Set)
        else:
            self.size = img_size
        self.mean_std = mean_std(self.bands, Set, bands)

    def __len__(self):
        return len(self.bands)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __getitem__(self, idx):
        band_name = self.bands[idx]
        mask_name = self.masks[idx]
        band_path = os.path.join(self.band_dir, band_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        # bandss = open_npy("bands", band_name, self.set)
        # mask = open_npy("masks", mask_name, self.set)
        # arrays_norm = [bandss[b] for b in self.band]
        Normilize = torchvision.transforms.Normalize(
            [self.mean_std[b][0] for b in self.band],
            [self.mean_std[b][1] for b in self.band],
        )
        # img = np.stack(arrays_norm, axis=2)
        img = torch.load(band_path)
        mask = torch.load(mask_path)

        if self.aug and self.training:
            mask = torch.squeeze(mask, 0)
            img = img.permute(1, 2, 0)
            img, mask = img_mask_augmentation(img.numpy(), mask.numpy())
            img, mask = torch.from_numpy(img).permute(2, 0, 1), torch.unsqueeze(
                torch.from_numpy(mask), 0
            )

        if self.norm:
            img_norm = Normilize(img)
        else:
            img_norm = img
        return img, mask
def get_data2(model_name, bands_selection, batch_size, img_size=-1, normalization=True):
    """
    Provide the user with the train, validation and test data loaders

    Parameters:
    bands_selection : List | Liste of the selected bands
    batch_size: int | number of images and masks in a batch
    img_size: int | output size. If < 0 average of the set
    normalization: bool | Is normalization is required

    Output:
    trainloader : DataLoader | DataLoader with training bands and masks
    valloader : DataLoader | Dataloader with validation bands and masks
    testloader : DataLoader | Dataloader with test bands and masks

    """

    train_data_set = SegmentationDataSet2(
        model_name=model_name,
        bands=bands_selection,
        Set="train",
        img_size=img_size,
        normalization=normalization,
        augmentation=True,
    )

    val_data_set = SegmentationDataSet2(
        model_name=model_name,
        bands=bands_selection,
        Set="val",
        img_size=img_size,
        normalization=normalization,
        augmentation=False,
    )
    test_data_set = SegmentationDataSet2(
        model_name=model_name,
        bands=bands_selection,
        Set="test",
        img_size=img_size,
        normalization=normalization,
        augmentation=False,
    )
    trainloader = DataLoader(train_data_set, batch_size=batch_size)
    valloader = DataLoader(val_data_set, batch_size=batch_size)
    testloader = DataLoader(test_data_set, batch_size=batch_size)
    return trainloader, valloader, testloader

def img_mask_augmentation(img, mask):
    func_geom = [
        A.augmentations.crops.transforms.RandomResizedCrop(
            height=img.shape[0],
            width=img.shape[1],
            scale=(0.01, 1.0),
            ratio=(0.5, 1.3333333333333333),
            interpolation=1,
            always_apply=False,
            p=1.0,
        ),
        A.augmentations.geometric.transforms.VerticalFlip(p=1),
        A.augmentations.geometric.transforms.Transpose(p=1),
        A.augmentations.geometric.transforms.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=45,
            interpolation=1,
            border_mode=4,
            value=None,
            mask_value=None,
            shift_limit_x=None,
            shift_limit_y=None,
            rotate_method="largest_box",
            always_apply=False,
            p=1,
        ),
        A.augmentations.geometric.transforms.OpticalDistortion(
            distort_limit=0.05,
            shift_limit=0.05,
            interpolation=1,
            border_mode=4,
            value=None,
            mask_value=None,
            always_apply=False,
            p=1,
        ),
        A.augmentations.geometric.transforms.HorizontalFlip(p=1),
        A.augmentations.geometric.transforms.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            interpolation=1,
            border_mode=4,
            value=None,
            mask_value=None,
            normalized=False,
            always_apply=False,
            p=1,
        ),
        A.augmentations.geometric.transforms.Flip(p=1),
        A.augmentations.geometric.transforms.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=150,
            interpolation=3,
            border_mode=4,
            value=None,
            mask_value=None,
            always_apply=False,
            approximate=False,
            same_dxdy=False,
            p=1,
        ),
        A.augmentations.geometric.transforms.Affine(
            scale=None,
            translate_percent=None,
            translate_px=None,
            rotate=None,
            shear=None,
            interpolation=1,
            mask_interpolation=0,
            cval=0,
            cval_mask=0,
            mode=0,
            fit_output=False,
            keep_ratio=False,
            rotate_method="largest_box",
            always_apply=False,
            p=1,
        ),
    ]
    func_nb = [
        A.augmentations.transforms.Superpixels(
            p_replace=0.1,
            n_segments=100,
            max_size=128,
            interpolation=1,
            always_apply=False,
            p=1,
        ),
        A.augmentations.transforms.Sharpen(
            alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=1
        ),
        A.augmentations.transforms.RandomGamma(
            gamma_limit=(80, 120), eps=None, always_apply=False, p=1
        ),
        A.augmentations.transforms.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            brightness_by_max=True,
            always_apply=False,
            p=1,
        ),
        A.augmentations.transforms.GaussNoise(
            var_limit=(0.04, 0.2), mean=0, per_channel=True, always_apply=False, p=1
        ),
        A.augmentations.transforms.InvertImg(p=1),
        A.augmentations.transforms.MultiplicativeNoise(
            multiplier=(0.9, 1.1),
            per_channel=False,
            elementwise=False,
            always_apply=False,
            p=1,
        ),
        A.augmentations.transforms.PixelDropout(
            dropout_prob=0.05,
            per_channel=False,
            drop_value=0,
            mask_drop_value=None,
            always_apply=False,
            p=1,
        ),
    ]
    func_1b_3b = [
        A.augmentations.transforms.ColorJitter(p=1),
        A.augmentations.transforms.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            always_apply=False,
            p=1,
        ),
    ]
    func_RGB = [
        A.augmentations.transforms.ToGray(p=1),
        A.augmentations.transforms.ToSepia(always_apply=False, p=1),
        A.augmentations.transforms.RandomFog(
            fog_coef_lower=0.3,
            fog_coef_upper=1,
            alpha_coef=0.08,
            always_apply=False,
            p=1,
        ),
        A.augmentations.transforms.RandomRain(
            slant_lower=-10,
            slant_upper=10,
            drop_length=20,
            drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=7,
            brightness_coefficient=0.7,
            rain_type=None,
            always_apply=False,
            p=1,
        ),
        A.augmentations.transforms.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            always_apply=False,
            p=1,
        ),
        A.augmentations.transforms.RandomSnow(
            snow_point_lower=0.1,
            snow_point_upper=0.3,
            brightness_coeff=2.5,
            always_apply=False,
            p=1,
        ),
        A.augmentations.transforms.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_lower=0,
            angle_upper=1,
            num_flare_circles_lower=1,
            num_flare_circles_upper=5,
            src_radius=400,
            src_color=(255, 255, 255),
            always_apply=False,
            p=1,
        ),
        A.augmentations.transforms.RGBShift(
            r_shift_limit=0.08,
            g_shift_limit=0.08,
            b_shift_limit=0.08,
            always_apply=False,
            p=1,
        ),
    ]
    if img.shape[-1] == 3:
        funct = func_nb + func_RGB + func_1b_3b
    elif img.shape[-1] == 1:
        funct = func_nb
    else:
        funct = func_nb
    vis_aug = np.random.choice(funct, size=1, replace=False)
    geom_aug = np.random.choice(func_geom, size=1, replace=False)
    pipeline = np.concatenate((vis_aug, geom_aug))
    transform = A.Compose(pipeline)
    aug = transform(image=img, mask=mask.astype(np.int32))
    img_aug, mask_aug = aug["image"], aug["mask"]
    # img_aug, mask_aug = np_uint8_to_float32(img_aug), np_uint8_to_float32(mask_aug)
    return img_aug, mask_aug