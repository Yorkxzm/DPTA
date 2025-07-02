import numpy as np
from torchvision import datasets, transforms
import os
import torch
import torchvision

def split_images_labels(imgs):
    ''' 
    split set.imgs in ImageFolder

    input: datasets.ImageFolder

    output: image and labels in np.array form
    '''
    images = []
    labels = []
    for img in imgs:
        images.append(img[0])
        labels.append(img[1])
    return np.array(images), np.array(labels)


class imgData(object):
    train_trans = []
    test_trans = []
    common_trans = []
    class_order = None

def build_transform(is_train,args=None,compose = False):
    #generate a customized torch transform for images
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        if compose == False:
            return transform
        else:
            return transforms.Compose(transform) #use compose to Composes several transform in a list to one transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())

    if compose == False:
        return t
    return transforms.Compose(t)

def build_transform_2(is_train, args):
    if is_train:        
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t

class iCIFAR100(imgData):
    def __init__(self, args=False):
        super().__init__()
        self.args = args
        self.use_path = False


        self.train_trans = build_transform(True)
        self.test_trans = build_transform(False)
        self.common_trans = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("../data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("../data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iImageNetR(imgData):
    use_path = True
    
    train_trans=build_transform(True, None)
    test_trans=build_transform(False, None)
    common_trans = [    ]


    class_order = np.arange(200).tolist()

    def download_data(self,train_dir='make_dataset/dataset/inr/train',test_dir='make_dataset/dataset/inr/test'):

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(imgData):
    use_path = True
    
    train_trans=build_transform(True, None)
    test_trans=build_transform(False, None)
    common_trans = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self,train_dir='make_dataset/dataset/ina/train',test_dir='make_dataset/dataset/ina/test'):

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class CUB(imgData):
    use_path = True
    
    train_trans=build_transform(True, None)
    test_trans=build_transform(False, None)
    common_trans = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self,train_dir='make_dataset/dataset/cub/train',test_dir='make_dataset/dataset/cub/test'):
        

        train_dset = datasets.ImageFolder(train_dir)#build a file index for all images in set and allocate integer labels for all classes
        test_dset = datasets.ImageFolder(test_dir)


        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class vtab(imgData):
    use_path = True
    
    train_trans=build_transform(True, None)

    test_trans=build_transform(False, None)

    common_trans = []

    class_order = np.arange(50).tolist() 

    def download_data(self,train_dir='make_dataset/dataset/vtab/train',test_dir='make_dataset/dataset/vtab/test'):
        # specify the folder of your dataset"

        train_dset = datasets.ImageFolder(train_dir)#build a file index for all images in set and allocate integer labels for all classes
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


#for stanford cars, it cannot be downloaded directly from torchvision now.
import pathlib
from typing import Any, Callable, Optional, Tuple, Union
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset
class StanfordCars():
    def __init__(self, root, split):

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")


        self._split = split
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        if self._check_exists() != 0:
            print(self._check_exists())
            raise RuntimeError(
                "Dataset not found. Try to manually download following the instructions in "
                "https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616."
            )

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.data: Any = []
        self.targets = []
        for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]:
            self.data.append(str(self._images_base_path / annotation["fname"]))
            self.targets.append(annotation["class"] - 1)
         

    def __len__(self) -> int:
        return len(self._samples)

    def _check_exists(self):
        if not (self._base_folder / "devkit").is_dir():
            return 1
        elif not self._annotations_mat_path.exists():
            return 2
        elif not self._images_base_path.is_dir():
            return 3
        return 0
    
class CARS(imgData):
    def __init__(self, args=False):
        super().__init__()
        self.args = args
        self.use_path = True


        self.train_trans = build_transform(True)
        self.test_trans = build_transform(False)
        self.common_trans = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(196).tolist()

    def download_data(self):
        train_dataset =  StanfordCars("make_dataset/dataset", split=True)
        test_dataset =  StanfordCars("make_dataset/dataset", split=False)
        
        self.train_data, self.train_targets = np.array(train_dataset.data), np.array(
            train_dataset.targets
        ).astype(int)
        print(len(self.train_data))
        print(len(self.train_targets))
        print(self.train_targets)
        print(self.train_targets.shape)
        self.test_data, self.test_targets = np.array(test_dataset.data), np.array(
            test_dataset.targets
        ).astype(int)