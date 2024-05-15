import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from sklearn.linear_model import LassoCV, LogisticRegression
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances, device):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth.to(device)

    def forward(self, X, device):
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(device)
        L2_distances = (torch.cdist(X, X) ** 2).to(device)
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances, device) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

class MMD(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel
        
    def forward(self, X, Y, device, weight = None):
        self.kernel.to(device)
        Y = Y.reshape(-1, X.shape[1])
        K = self.kernel(torch.vstack([X, Y]), device)
        X_size = X.shape[0]
        if weight is not None:
            xweight = torch.abs(weight).to(device)
            # xweight = xweight / xweight.sum()
            yweight = torch.ones(Y.shape[0])
            # yweight = yweight / yweight.sum()
            yweight = yweight.to(device)
            XXweight = torch.outer(xweight, xweight)
            XYweight = torch.outer(xweight, yweight)
            YYweight = torch.outer(yweight, yweight)

            XX = (K[:X_size, :X_size] * XXweight).mean()
            XY = (K[:X_size, X_size:] * XYweight).mean()
            YY = (K[X_size:, X_size:] * YYweight).mean()
        else:
            XX = K[:X_size, :X_size].mean()
            XY = K[:X_size, X_size:].mean()
            YY = K[X_size:, X_size:].mean()

        return XX - 2 * XY + YY

def mmdWeight(X, y, num_steps, lr, lambdda, selected_n_list = None):
    n, p = X.shape
    if selected_n_list is None:
        weight = torch.ones(n, dtype=float)
    else:
        weight = torch.ones(len(selected_n_list), dtype=float)
    weight.requires_grad = True
    optimizer = optim.Adam([weight,], lr = lr)
    mmd = MMD()
    selected_n_list = torch.tensor(selected_n_list)
    for i in range(num_steps):
        optimizer.zero_grad()
        if selected_n_list is not None:
            sample_wise_weight = weight.repeat_interleave(selected_n_list)
            dis = mmd(X, y, device, weight = sample_wise_weight)
        else:
            dis = mmd(X, y, device, weight = weight)
        # lasso
        loss = dis + lambdda * torch.norm(weight, p=1)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0 or i + 1 == num_steps :
            print(f"iter {i}  dis: {dis}  loss: {loss}")
    return abs(weight).cpu().detach().numpy()

def cosine_similarity(vec1, vec2):
    dot_product = torch.dot(vec1.view(-1).float(), vec2.view(-1).float())
    norm_vec1 = torch.linalg.norm(vec1)
    norm_vec2 = torch.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def cosine_dis(X, weight, y):
    weighted_center = torch.mm(X, abs(weight).unsqueeze(1)).reshape(-1)
    dis = 1 - cosine_similarity(weighted_center, y)
    return dis 

def cosineDis(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return 1 - dot_product / (norm_vec1 * norm_vec2)

def cosineWeight(X, y, num_steps, lr, lambdda):
    n, p = X.shape
    weight = torch.ones(p, dtype=float)
    weight.requires_grad = True
    optimizer = optim.Adam([weight,], lr = lr)
    for i in range(num_steps):
        optimizer.zero_grad()
        dis = cosine_dis(X, weight, y)
        
        # lasso
        loss = dis + lambdda * torch.norm(weight, p=1)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0 or i + 1 == num_steps :
            print(f"iter {i}  dis: {dis}  loss: {loss}")
    return abs(weight).cpu().detach().numpy()

def mseWeight(X, y, args):
    model = LassoCV(max_iter = 5000, random_state = args.seed)
    model.fit(X, y)
    return np.abs(model.coef_)

def ERM_train_test(model, X, y, X_t):
    model.fit(X, y)
    return model.predict(X_t)

def pseudo_train_test(model, X, y, visible_X_t, X_t):
    model.fit(X, y)
    bar = 0.8
    p = model.predict_proba(visible_X_t)
    pseudo_x, pseudo_y = [], []
    for i, x in enumerate(visible_X_t):
        if p[i][0] > bar:
            pseudo_x.append(x)
            pseudo_y.append(0)
        elif p[i][1] > bar:
            pseudo_x.append(x)
            pseudo_y.append(1)
    pseudo_x = np.array(pseudo_x)
    pseudo_y = np.array(pseudo_y)
    X = np.concatenate([X, pseudo_x], axis = 0)
    Y = np.concatenate([y, pseudo_y], axis = 0)
    model.fit(X, Y)

    return model.predict(X_t)

def reweight_train_test(model, total_X, y, visible_X_t, X_t):
    env_classifer = LogisticRegression(max_iter = 3000)
    X = np.concatenate([total_X, visible_X_t], axis = 0)
    Y = np.concatenate([np.zeros(len(total_X)), np.ones(len(visible_X_t))], axis = 0)

    env_classifer.fit(X, Y)
    p = env_classifer.predict_proba(total_X)
    weight = np.divide(p[:, 1], p[:, 0] + 1e10)
    weight /= np.sum(weight)
    weight += 1
    model.fit(total_X, y, sample_weight=weight)
    return model.predict(X_t)


