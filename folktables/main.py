import argparse
import torch
import numpy as np
import os
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils import *
states_list = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]
def get_args():
    parser = argparse.ArgumentParser(description="Script to launch sample reweighting experiments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--task", choices=["ACSIncome", "ACSPublicCoverage", "ACSEmployment", "ACSMobility", "ACSTravelTime"], default="ACSIncome")
    parser.add_argument("--test_state", choices=states_list, default='OH')
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambdda", type=float, default=0.000001)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dis", type=str, default="mmd", choices=["mmd", "cosine", "mse"])
    return parser.parse_args()

def setup_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def top(n_list, top_id, all_X, all_y):
    """
    Unifromly pick samples from each sources in "top_id"
    """
    while(np.sum(n_list) < args.T):
        for id in top_id:
            n_list[id] += 1
            if np.sum(n_list) == args.T:
                break
    total_X, total_y = np.array([]), np.array([])
    for i, state in enumerate(states_list):
        X = all_X[i][:n_list[i]]
        y = all_y[i][:n_list[i]]
        if i == 0:
            total_X = X
            total_y = y
        else:
            total_X = np.concatenate([total_X, X], axis = 0)
            total_y = np.concatenate([total_y, y], axis = 0)
    return total_X, total_y

def DAA(n_list, all_X, all_y, top10_id, mean_t, visible_X_t,  n_per_round, args):
    """
    Only do data acquisition in selected 10 envs start from the second round,

    You may change this "top10_id" to any "topk_id"
    """
    mean_s, X_s = [], []
    selected_n_list = []
    for id in top10_id:
        X = all_X[id][:n_list[id]]
        y = all_y[id][:n_list[id]]
        mean_s.append(np.mean(X, axis = 0))
        X_s.append(X)
        selected_n_list.append(n_list[id])

    X_s = np.concatenate(X_s)
    for j in range(1, args.k):
        print(f"Round {j}:")
        mean_s = np.array(mean_s).T
        if args.dis == "cosine":
            weights = cosineWeight(torch.tensor(mean_s), torch.tensor(mean_t), num_steps=3000, lr = args.lr, lambdda = args.lambdda)
        elif args.dis == "mmd":
            # simplified version
            # weights = mmdWeight(torch.tensor(mean_s).T, torch.tensor(mean_t), num_steps=3000, lr = args.lr, lambdda = args.lambdda)
            
            # completed version
            weights = mmdWeight(torch.tensor(X_s), torch.tensor(visible_X_t), num_steps=2000, lr = args.lr, lambdda = args.lambdda, selected_n_list = selected_n_list)
        elif args.dis == "mse":
            weights = mseWeight(mean_s, mean_t, args)
        else:
            raise NotImplementedError
        
        # Use weight to collect more data

        total = np.sum(weights)
        fnumbers = [(n_per_round * weights[i] / total) for i in range(len(weights))]
        sorted_indices = sorted(range(len(fnumbers)), key=lambda k: fnumbers[k] % 1, reverse=True)
        delta = [int(v) for v in fnumbers]
        i = 0
        while(np.sum(delta) < n_per_round):
            j = sorted_indices[i]
            delta[j] += 1
            i += 1
        X_s = []
        for i, id in enumerate(top10_id):
            n_list[id] += delta[i]
            selected_n_list[i] = n_list[id]
            X = all_X[id][:n_list[id]]
            X_s.append(X)
        X_s = np.concatenate(X_s)
        mean_s = [np.mean(all_X[i][:n_list[i]], axis = 0) for i in top10_id]

    X = np.concatenate([all_X[i][:n_list[i]] for i in range(len(all_X))], axis = 0)
    y = np.concatenate([all_y[i][:n_list[i]] for i in range(len(all_y))], axis = 0)

    return X, y, n_list

def main(args):
    model = LogisticRegression(max_iter = 3000, random_state = args.seed) # all methods share the same linear model structure
    dis_dict = {}
    random_indices = np.random.choice(len(X_t), size=args.n_test, replace=False)
    visible_X_t = X_t[random_indices]
    mean_t = np.mean(visible_X_t, axis = 0)

    all_X, all_y = [], []
    n_per_round = args.T // args.k
    n_list = [n_per_round // 49] * 49  # first round: uniformly collect samples from each domain
    id_dict = {}
    i = 0
    while(np.sum(n_list) < n_per_round):
        n_list[i] += 1
        i += 1
    for i, state in enumerate(states_list):
        X = np.load(f"data/{args.task}/{state}_X.npy")
        y = np.load(f"data/{args.task}/{state}_y.npy")
        if args.T <= len(X):
            replace = False
        else:
            replace = True
        random_indices = np.random.choice(len(X), size=args.T, replace=replace)
        X = X[random_indices]
        y = y[random_indices]
        all_X.append(X)
        all_y.append(y)
        id_dict[state] = i
        mean_point = np.mean(X[:n_list[i]], axis = 0)
        if args.dis == "cosine":
            dis_dict[state] = cosineDis(mean_point, mean_t)
        elif args.dis == "mmd":
            mmd = MMD()
            dis_dict[state] = mmd(torch.tensor(X[:n_list[i]]), torch.tensor(visible_X_t), device = "cpu")
        elif args.dis == "mse":
            dis_dict[state] = np.sum((mean_point - mean_t) ** 2)
        else:
            raise NotImplementedError
        
    dis_dict = sorted(dis_dict.items(), key=lambda x: x[1])
    top10 = [x[0] for x in dis_dict[:10]]
    top10_id = [id_dict[x] for x in top10]
    top3 = [x[0] for x in dis_dict[:3]]
    top3_id = [id_dict[x] for x in top3]
    
    # all source envs
    total_X, total_y = top(n_list.copy(), range(len(states_list)), all_X, all_y)

    yp = ERM_train_test(model, total_X, total_y, X_t) # ERM
    acc_uni = accuracy_score(yp, y_t)
    yp = pseudo_train_test(model, total_X, total_y, visible_X_t, X_t) # pseudo_labeling
    pacc_uni = accuracy_score(yp, y_t)
    yp = reweight_train_test(model, total_X, total_y, visible_X_t, X_t) #sample_reweighting
    wacc_uni = accuracy_score(yp, y_t)

    # 10 envs
    total_X, total_y = top(n_list.copy(), top10_id, all_X, all_y)

    yp = ERM_train_test(model, total_X, total_y, X_t) # ERM
    acc_10 = accuracy_score(yp, y_t)
    yp = pseudo_train_test(model, total_X, total_y, visible_X_t, X_t) # pseudo_labeling
    pacc_10 = accuracy_score(yp, y_t)
    yp = reweight_train_test(model, total_X, total_y, visible_X_t, X_t) #sample_reweighting
    wacc_10 = accuracy_score(yp, y_t)

    # 3 envs
    total_X, total_y = top(n_list.copy(), top3_id, all_X, all_y)

    yp = ERM_train_test(model, total_X, total_y, X_t) # ERM
    acc_3 = accuracy_score(yp, y_t)
    yp = pseudo_train_test(model, total_X, total_y, visible_X_t, X_t) # pseudo_labeling
    pacc_3 = accuracy_score(yp, y_t)
    yp = reweight_train_test(model, total_X, total_y, visible_X_t, X_t) #sample_reweighting
    wacc_3 = accuracy_score(yp, y_t)

    # DAA
    X, y, n_list = DAA(n_list.copy(), all_X, all_y, top10_id, mean_t, visible_X_t, n_per_round, args)
    
    yp = ERM_train_test(model, X, y, X_t) # ERM
    acc = accuracy_score(yp, y_t)
    yp = pseudo_train_test(model, X, y, visible_X_t, X_t) # pseudo_labeling
    pacc  = accuracy_score(yp, y_t)
    yp = reweight_train_test(model, X, y, visible_X_t, X_t) #sample_reweighting
    wacc = accuracy_score(yp, y_t)

    return [acc, pacc, wacc, acc_3, pacc_3, wacc_3, acc_10, pacc_10, wacc_10, acc_uni, pacc_uni, wacc_uni]

if __name__ == "__main__":
    args = get_args()
    setup_seed(args)
    test_state = args.test_state
    states_list = list(set(states_list) - set([test_state])) # exclude the test state
    # prepare test data
    X_t = np.load(f"data/{args.task}/{test_state}_X.npy")
    y_t = np.load(f"data/{args.task}/{test_state}_y.npy")
    g_t = np.load(f"data/{args.task}/{test_state}_g.npy")

    lists = []
    repeat_times = 3
    for i in range(repeat_times):
        print(f"test_state{args.test_state}, repeat{i}")
        acc_list = main(args)
        lists.append(acc_list)
        
    acc_list = np.mean(lists, axis=0)
    dir = f"results/{args.task}_{args.test_state}"
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # output order: 
    # ours, top3, top10, uniform(top49)
    with open(f"{dir}/erm.txt", "a") as f:
        f.write(f"{args.dis}, k={args.k}, T={args.T}\n")
        f.write(f"{acc_list[0]}, {acc_list[3]}, {acc_list[6]}, {acc_list[9]} \n")

    with open(f"{dir}/self_supervise.txt", "a") as f:
        f.write(f"{args.dis}, k={args.k}, T={args.T}\n")
        f.write(f"{acc_list[1]}, {acc_list[4]}, {acc_list[7]}, {acc_list[10]} \n")

    with open(f"{dir}/reweight.txt", "a") as f:
        f.write(f"{args.dis}, k={args.k}, T={args.T}\n")
        f.write(f"{acc_list[2]}, {acc_list[5]}, {acc_list[8]}, {acc_list[11]} \n")

