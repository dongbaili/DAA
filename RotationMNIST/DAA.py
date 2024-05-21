import argparse
import random
import time
from domainbed.datasets import _dataset_info, StandardDataset, get_train_transformer
import numpy as np
import torch
import torch.utils.data
from domainbed.lib import misc

from utils import *

def load_args():
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--txtdir', type=str, default="domainbed/txtlist")
    parser.add_argument('--dataset', type=str, default="RMnist")
    parser.add_argument("--source", nargs='+')
    parser.add_argument("--target", nargs="+")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n', type=int, help="Total budget we have")
    parser.add_argument('--k', type=int, help="number of selection rounds")
    parser.add_argument('--visible_n_test', type=int, help="number of visible samples from test domain")
    parser.add_argument('--dis', type=str, choices=['mse', 'cosine', 'mmd'], help="distance metric used to train Lasso")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    start = time.time()
    args = load_args()

    misc.setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # prepare all X_s
    total_names, total_labels = [], []
    for source in args.source:
        dir = f"{args.txtdir}/{args.dataset}/{source}.txt"
        names, labels = _dataset_info(dir)
        total_names.extend(names)
        total_labels.extend(labels)
    img_transformer = get_train_transformer()
    Dataset = StandardDataset(total_names, total_labels, img_transformer)
    X_s, y_s= [], []
    for i in range(len(Dataset)):
        X_s.append(Dataset.get_image(i).to(device))
        y_s.append(Dataset.labels[i])
    X_s = torch.stack(X_s).reshape(len(X_s), -1)

    # prepare all X_t
    dir = f"{args.txtdir}/{args.dataset}/{args.target[0]}.txt"
    names, labels = _dataset_info(dir)
    Dataset = StandardDataset(names, labels, img_transformer)
    X_t = []
    for i in range(len(Dataset)):
        X_t.append(Dataset.get_image(i).to(device))
    
    X_t = torch.stack(X_t).reshape(len(X_t), -1)
    y_t = labels
    X = torch.cat([X_s, X_t], dim = 0)
    Y = torch.cat([torch.tensor(y_s), torch.tensor(y_t)], dim = 0)

    # DAA algorithm
    after_loading = time.time()
    n_per_step = args.n / args.k 
    source_envs = args.source
    target = torch.mean(X[-args.visible_n_test:], axis=0)
    dis = args.dis

    indexs = [0, args.n, 2 * args.n, 3 * args.n]

    # initial: randomly pick samples from each sources
    ns = [int(n_per_step // len(source_envs))] * len(source_envs)
    i = 0
    while(np.sum(ns) < n_per_step):
        ns[i] += 1
        i += 1
    for j in range(1, args.k):
        source = [torch.mean(X[i:i + ni], axis=0) for (i,ni) in zip(indexs, ns)]
        print(f"Round {j}:")
        if dis == "mse": 
            alphas = None
            # train domain weight by lasso
            weights = abs(mseSklearn(source, target.detach().cpu().numpy(), alphas))
        elif dis == "cosine": 
            source = torch.stack(source, dim=1)
            # train domain weight by lasso
            weights, dis_list = cosineTorch(source.detach().clone(), target.detach().clone(), num_steps=1500, lr = 0.001, lambdda = 0.01)
        elif dis == "mmd":
            all_s = torch.vstack([X[i:i + ni] for (i,ni) in zip(indexs, ns)])
            all_t = X[-args.visible_n_test:].clone().detach()
            source = torch.stack(source, dim = 1)
            
            # simplified version
            # weights, dis_list = mmdTorch(source.detach().clone().T, target.detach().clone(), num_steps=1000, lr = 0.01, lambdda = 0.001)

            # completed version
            weights, dis_list = mmdTorch(all_s, all_t, num_steps=1000, lr = 0.01, lambdda = 0.001, selected_n_list = ns)
        # update ns based on the weight
        total = np.sum(weights)
        fnumbers = [(n_per_step * weights[i] / total) for i in range(len(weights))]
        sorted_indices = sorted(range(len(fnumbers)), key=lambda k: fnumbers[k] % 1)
        delta = [int(v) for v in fnumbers]
        i = 0
        while(np.sum(delta) < n_per_step):
            j = sorted_indices[i]
            delta[j] += 1
            i += 1
        ns = [ns[i] + delta[i] for i in range(len(ns))]
        print("delta:", delta, np.sum(delta))
        # print("weights: ", weights)

    end = time.time()
    # record number of samples from each source domain
    with open("results/samples.txt", 'a') as f:
        f.write(f"dis = {args.dis}, n = {args.n}, K = {args.k}, {ns}\n")

    print("-------------------------------")
    print("Final results: ")
    print("total running time: ", end - start)
    print("DAA algorithm running time: ", end - after_loading)
    
    # generate .txt file
    total_lines = []
    for source,ni in zip(args.source, ns):
        print(source, ni)
        with open(f"domainbed/txtlist/RMnist/{source}.txt", 'r') as origin_f:
            lines = origin_f.readlines()[:ni]
            total_lines.extend(lines)

    random.shuffle(total_lines)
    with open(f"domainbed/txtlist/RMnist/DAA_{dis}.txt", 'w') as f:
        for line in total_lines:
            f.write(line)
    