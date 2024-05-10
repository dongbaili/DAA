# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data as data
from torchvision import transforms
from PIL import Image, ImageFile
from os.path import join
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.networks import MNIST_CNN
# from domainbed.algorithms import get_algorithm_class
import numpy as np


ImageFile.LOAD_TRUNCATED_IMAGES = True

num_classes_dict = {
    "PACS": 7,
    "VLCS": 5,
    "OfficeHome": 65,
    "DomainNet": 345,
    "TerraInc": 10,
    "NICO": 60,
    "RMnist": 10,
    "FMOW": 7,    # time and country
    # "FMOW": 24,     # time only
}

checkpoint_step_dict = {
    "PACS": 300,
    "VLCS": 300,
    "OfficeHome": 300,
    "DomainNet": 1000,
    "TerraInc": 300,
    "NICO": 600,
    "RMnist": 300,
    "FMOW": 300,
}


train_steps_dict = {
    "PACS": 5000,
    "VLCS": 5000,
    "OfficeHome": 5000,
    "TerraInc": 5000,
    "DomainNet": 15000,
    "NICO": 10000,
    "RMnist": 5000,
    "FMOW" : 5000,
}

def _dataset_info(txt_file):
    with open(txt_file, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.strip().split(' ')
        file_names.append(' '.join(row[:-1]))
        labels.append(int(row[-1]))

    return file_names, labels


class StandardDataset(data.Dataset):
    def __init__(self, names, labels, img_transformer=None):
        self.names = names
        self.labels = labels

        self.N = len(self.names)
        self._image_transformer = img_transformer
    
    def get_image(self, index):
        img = Image.open(self.names[index]).convert('RGB')
        return self._image_transformer(img)
    
    def get_raw_image(self, index):
        mnist_array = np.array(Image.open(self.names[index]))

        # 将像素值限制在0和255之间
        mnist_array = np.clip(mnist_array, 0, 255)

        # 转换为uint8类型
        mnist_array = mnist_array.astype(np.uint8)

        return mnist_array
    
    def get_picture(self, index):
        return Image.open(self.names[index]).convert('RGB')
    
    def __getitem__(self, index):
        img = self.get_image(index)
        return img, int(self.labels[index])

    def __len__(self):
        return len(self.names)

    

def get_train_transformer(): # hard-coded
    return transforms.Compose([
        transforms.RandomResizedCrop(28, scale=(0.7, 1.0)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        # transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_val_transformer(): # hard-coded
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
def get_data_transformer(mode):
    assert mode in ["train", "eval"]
    if mode == "train":
        return get_train_transformer()
    else:
        return get_val_transformer()

def get_dataloader(txtdir, dataset, domain, batch_size, mode="train", split=True, holdout_fraction=0.2, num_workers=8, seed=1):
    assert mode in ["train", "eval"]
    loader_func = {"train": InfiniteDataLoader, "eval": FastDataLoader}
    names, labels = _dataset_info(join(txtdir, dataset, "%s.txt"%domain))
    if split:
        idxs = np.arange(len(names))
        np.random.RandomState(seed).shuffle(idxs)
        mid = int(len(idxs)*(1-holdout_fraction))
        idxs_dict = {"train": idxs[:mid], "eval": idxs[mid:]}
        loader_func = {"train": InfiniteDataLoader, "eval": FastDataLoader}
        loader_dict = dict()
        for key, idxs in idxs_dict.items():
            names_split = [names[idx] for idx in idxs]
            labels_split = [labels[idx] for idx in idxs]
            img_tr = get_data_transformer(key)
            dataset_split = StandardDataset(names_split, labels_split, img_tr)
            loader = loader_func[key](dataset=dataset_split, batch_size=batch_size, num_workers=num_workers)
            loader_dict[key] = loader
        return loader_dict
    else:
        img_tr = get_data_transformer(mode)
        curDataset = StandardDataset(names, labels, img_tr)
        loader = loader_func[mode](dataset=curDataset, batch_size=batch_size, num_workers=num_workers)
        return loader

def get_mix_dataloader(txtdir, dataset, domains, phase, batch_size, num_workers=8):
    assert phase == "train"
    img_tr = get_train_transformer()
    concat_list = []
    for domain in domains:
        names, labels = _dataset_info(join(txtdir, dataset, "%s_%s.txt"%(domain, phase)))
        curDataset = StandardDataset(names, labels, img_tr)
        concat_list.append(curDataset)
    finalDataset = data.ConcatDataset(concat_list)
    loader = InfiniteDataLoader(dataset=finalDataset, weights=None, batch_size=batch_size, num_workers=num_workers)
    return loader

def get_central(txtdir, dataset, domain, mode="eval"):
    names, labels = _dataset_info(join(txtdir, dataset, "%s.txt"%domain))
    img_tr = get_data_transformer(mode)
    curDataset = StandardDataset(names, labels, img_tr)
    X = []
    for i in range(len(names)):
        x = curDataset.get_image(i)
        X.append(x)
    X = np.mean(X, axis = 0)
    return X

def get_dataloaders(args, hparams, logger):
    # set up dataloaders
    if args.mix:
        # TO FIX
        trainloaders = [get_mix_dataloader(args.txtdir, args.dataset, args.source, "train", hparams["batch_size"]),]
        logger.info("Train size: %d" % len(trainloaders[0].dataset))
    else:
        trainloaders = []
        valloaders = []
        for domain in args.source:
            loader_dict = get_dataloader(args.txtdir, args.dataset, domain, hparams["batch_size"], mode="train", split=True, holdout_fraction=args.holdout_fraction, seed=args.data_seed)
            trainloaders.append(loader_dict["train"])
            valloaders.append(loader_dict["eval"])
    testloaders = [get_dataloader(args.txtdir, args.dataset, domain, hparams["batch_size"], mode="eval", split=False) for domain in args.target]

    for index, domain in enumerate(args.source):
        logger.info("Train %s size: %d" % (domain, len(trainloaders[index].dataset)))
    for index, domain in enumerate(args.source):
        logger.info("Val %s size: %d" % (domain, len(valloaders[index].dataset)))
    for index, domain in enumerate(args.target):
        logger.info("Test %s size: %d" % (domain, len(testloaders[index].dataset)))

    return trainloaders, valloaders, testloaders