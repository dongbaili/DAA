# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import wide_resnet
from domainbed.lib import vits_moco
from domainbed.lib.ms_resnet import ms_resnet18, ms_resnet50
from domainbed.lib.simclr_resnet import get_resnet, name_to_params
import copy
import os
from collections import OrderedDict

import clip
from copy import deepcopy

weights_path = {
    "vits16": {
        "MoCo-v3": "/home/hanyu/Pretrained_Weights/moco_v3_vit-s-300ep.pth.tar"
    },
    "vitb16": {
        "MoCo-v3": "/home/hanyu/Pretrained_Weights/moco_v3_vit-b-300ep.pth.tar"        
    },
    "resnet50": {
        "MoCo": "/home/hanyu/Pretrained_Weights/moco_v1_200ep_pretrain.pth.tar",
        "MoCo-v2": "/home/hanyu/Pretrained_Weights/moco_v2_800ep_pretrain.pth.tar",
        "MoCo-v2-200": "/home/hanyu/Pretrained_Weights/moco_v2_200ep_pretrain.pth.tar",
        "SimCLR-v2": "/home/hanyu/Pretrained_Weights/simclr_v2_r50_1x_sk0.pth",
        "Triplet": "/home/hanyu/Pretrained_Weights/triplet_release_ep200.pth",
        "CaCo": "/home/hanyu/Pretrained_Weights/caco_single_4096_200ep.pth.tar",
        "SimSiam": "/home/hanyu/Pretrained_Weights/simsiam_checkpoint_0099.pth.tar",
        "SwAV": "/home/hanyu/Pretrained_Weights/swav_400ep_2x224_pretrain.pth.tar",
        "Barlow-Twins": "/home/hanyu/Pretrained_Weights/barlow_resnet50.pth",
        "InfoMin": "/home/hanyu/Pretrained_Weights/InfoMin_200.pth",
        "SimCLR": "/home/hanyu/Pretrained_Weights/simclr_checkpoint_0040.pth.tar"        
    }
}

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        do_pretrain = False if hparams["pretrain"] == "None" else True
        if hparams["arch"] == 'resnet18':
            if hparams['do_ms']:
                self.network = ms_resnet18(pretrained=do_pretrain, hparams=hparams)
            else: 
                self.network = torchvision.models.resnet18(pretrained=do_pretrain)
            self.n_outputs = 512
        elif hparams["arch"] == 'resnet50':
            if hparams['do_ms']:
                self.network = ms_resnet50(pretrained=do_pretrain, hparams=hparams)
            else:
                if hparams["pretrain"] == "Supervised":
                    self.network = torchvision.models.resnet50(pretrained=True)
                elif hparams["pretrain"] == "None":
                    self.network = torchvision.models.resnet50(pretrained=False)
                elif hparams["pretrain"] in ["MoCo", "MoCo-v2", "MoCo-v2-200", "CaCo"]:
                    self.network = torchvision.models.resnet50(pretrained=False)
                    assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                    checkpoint = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")
                    # rename moco pre-trained keys
                    state_dict = checkpoint["state_dict"]
                    for k in list(state_dict.keys()):
                        # retain only encoder_q up to before the embedding layer
                        if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
                            # remove prefix
                            state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
                        # delete renamed or unused k
                        del state_dict[k]

                    msg = self.network.load_state_dict(state_dict, strict=False)
                    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
                    print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))
                elif hparams["pretrain"] == "SimSiam":
                    self.network = torchvision.models.resnet50(pretrained=False)
                    assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                    checkpoint = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")
                    # rename moco pre-trained keys
                    state_dict = checkpoint["state_dict"]
                    for k in list(state_dict.keys()):
                        # retain only encoder_q up to before the embedding layer
                        if k.startswith("module.encoder") and not k.startswith("module.encoder.fc"):
                            # remove prefix
                            state_dict[k[len("module.encoder.") :]] = state_dict[k]
                        # delete renamed or unused k
                        del state_dict[k]

                    msg = self.network.load_state_dict(state_dict, strict=False)
                    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
                    print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))
                elif hparams["pretrain"] == "SimCLR":
                    self.network = torchvision.models.resnet50(pretrained=False)
                    assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                    checkpoint = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")   
                    state_dict = checkpoint["state_dict"]
                    for k in list(state_dict.keys()):
                        if k.startswith('backbone.'):
                            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                            # remove prefix
                                state_dict[k[len("backbone."):]] = state_dict[k]
                        del state_dict[k] 
                    msg = self.network.load_state_dict(state_dict, strict=False)
                    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}                        
                    print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))               
                elif hparams["pretrain"] == "SimCLR-v2":
                    assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                    checkpoint = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")   
                    self.network, _ = get_resnet(*name_to_params(weights_path[hparams["arch"]][hparams["pretrain"]].split("/")[-1]))
                    self.network.load_state_dict(checkpoint['resnet'])
                elif hparams["pretrain"] == "Triplet":
                    self.network = torchvision.models.resnet50(pretrained=False)
                    assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                    checkpoint = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")
                    state_dict = checkpoint["state_dict"]
                    msg = self.network.load_state_dict(state_dict, strict=False)
                    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
                    print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))
                    # print(state_dict.keys())
                    # print(self.network.state_dict().keys())
                elif hparams["pretrain"] == "SwAV":
                    self.network = torchvision.models.resnet50(pretrained=False)
                    assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                    state_dict = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")
                    # remove prefixe "module."
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                    for k, v in self.network.state_dict().items():
                        if k not in list(state_dict):
                            print('key "{}" could not be found in provided state dict'.format(k))
                        elif state_dict[k].shape != v.shape:
                            print('key "{}" is of different shape in model and provided state dict'.format(k))
                            state_dict[k] = v
                    msg = self.network.load_state_dict(state_dict, strict=False)
                    print("Load pretrained model with msg: {}".format(msg))
                elif hparams["pretrain"] == "Barlow-Twins":
                    self.network = torchvision.models.resnet50(pretrained=False)
                    assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                    state_dict = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")
                    missing_keys, unexpected_keys = self.network.load_state_dict(state_dict, strict=False)
                    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
                    print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))
                elif hparams["pretrain"] == "InfoMin":
                    self.network = torchvision.models.resnet50(pretrained=False)
                    assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                    state_dict = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")['model']
                    encoder_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        k = k.replace('module.', '')
                        if 'encoder' in k:
                            k = k.replace('encoder.', '')
                            encoder_state_dict[k] = v
                    msg = self.network.load_state_dict(encoder_state_dict, strict=False)
                    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
                    print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))
                elif hparams["pretrain"] == "CLIP":
                    self.network = torchvision.models.resnet50(pretrained=False)
                    model, preprocess = clip.load("RN50", device="cpu")
                    self.network = deepcopy(model.visual)
                    # msg = self.network.load_state_dict(model.visual.state_dict(), strict=False)
                    # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
                    print("=> loaded pre-trained model '{}'".format("CLIP"))
                    # set1 = set(model.visual.state_dict().keys())
                    # set2 = set(torchvision.models.resnet50(pretrained=False).state_dict().keys())
                    # print(len(set1))
                    # print(len(set2))
                    # print(set1&set2)
                    # print(set1-set2)
                    # print(set2-set1)
                    # assert False
                else:
                    raise NotImplementedError
            if hparams["pretrain"] == "CLIP":
                self.n_outputs = 1024
            else:
                self.n_outputs = 2048
        elif hparams["arch"] == "vitb16":
            if hparams["pretrain"] == "Supervised":
                self.network = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
            elif hparams["pretrain"] == "MoCo-v3":
                self.network = vits_moco.vit_base()
                linear_keyword = 'head'
                state_dict = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")['state_dict']
                for k in list(state_dict.keys()):
                    # retain only base_encoder up to before the embedding layer
                    if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                        # remove prefix
                        state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                msg = self.network.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
                del self.network.head
                self.network.head = Identity()
                print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))
            else:
                raise NotImplementedError
            self.n_outputs = 768
        elif hparams["arch"] == "vits16":
            if hparams["pretrain"] == "MoCo-v3":
                self.network = vits_moco.vit_small()
                linear_keyword = 'head'
                state_dict = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")['state_dict']
                for k in list(state_dict.keys()):
                    # retain only base_encoder up to before the embedding layer
                    if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                        # remove prefix
                        state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                msg = self.network.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
                del self.network.head
                self.network.head = Identity()
                print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))
                self.n_outputs = 384
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        if hparams["pretrain"] == "CLIP":
            pass
        elif "resnet" in hparams["arch"]:
            del self.network.fc
            self.network.fc = Identity()
        elif "vit" in hparams["arch"]:
            if "heads" in dir(self.network):
                del self.network.heads
                self.network.heads = Identity()
        else:
            raise NotImplementedError
        if hparams["pretrain"] != "None":
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.hparams["pretrain"] != "None":
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)
