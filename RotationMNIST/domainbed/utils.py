import torch
import torch.nn as nn

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

class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y, device, weight = None):
        self.kernel.to(device)
        K = self.kernel(torch.vstack([X, Y]), device)
        X_size = X.shape[0]
        if weight is not None:
            # x上的weight: 取绝对值 + 归一化
            # weight = torch.abs(weight)
            # weight /= torch.sum(weight)
            # y上的weight：均匀，归一
            yweight = torch.ones(Y.shape[0]) / Y.shape[0]
            yweight = yweight.to(device)
            # 矩阵权重：向量外积
            XXweight = torch.outer(weight, weight)
            XYweight = torch.outer(weight, yweight)
            YYweight = torch.outer(yweight, yweight)

            XX = (K[:X_size, :X_size] * XXweight).mean()
            XY = (K[:X_size, X_size:] * XYweight).mean()
            YY = (K[X_size:, X_size:] * YYweight).mean()
        else:
            XX = K[:X_size, :X_size].mean()
            XY = K[:X_size, X_size:].mean()
            YY = K[X_size:, X_size:].mean()

        return XX - 2 * XY + YY