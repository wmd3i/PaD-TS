import torch
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch
import numpy as np




def cross_correlation_distribution(data):
    def get_lower_triangular_indices_no_diag(n):
        indices = torch.tril_indices(n, n).to(data).long()
        indices_without_diagonal = (indices[0] != indices[1]).nonzero(as_tuple=True)
        return indices[0][indices_without_diagonal], indices[1][indices_without_diagonal]
        
    index = get_lower_triangular_indices_no_diag(data.shape[2])
    toreturn = []
    for i in range(data.shape[0]):
        corr_matrix = torch.corrcoef(data[i].T)
        toreturn.append(corr_matrix[index])
    toreturn = torch.stack(toreturn, dim=0).to(data)
    
    # Replace inf and NaN values with 0
    toreturn = torch.where(torch.isinf(toreturn) | torch.isnan(toreturn), torch.tensor(0.0), toreturn)
    return toreturn.float()


 # From https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy
def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(x),
                  torch.zeros(xx.shape).to(x),
                  torch.zeros(xx.shape).to(x))

    if kernel == "multiscale":
        bandwidth_range = [0.01,0.05,0.1,0.2,0.5]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [0.01,0.05,0.1,0.2,0.5,0.7,1.0,1.5,2.0]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)


def BMMD(x, y, kernel):
    """Empirical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P : Shape (Feature, Sample, 1)
        y: second sample, distribution Q : Shape (Feature, Sample, 1)
        kernel: kernel type such as "multiscale" or "rbf"
    """
    # Compute matrix products
    xx = torch.bmm(x, x.transpose(1, 2))
    yy = torch.bmm(y, y.transpose(1, 2))
    zz = torch.bmm(x, y.transpose(1, 2))

    # Compute diagonal matrices
    rx = torch.diagonal(xx, dim1=1, dim2=2).unsqueeze(2).expand_as(xx).transpose(1,2)
    ry = torch.diagonal(yy, dim1=1, dim2=2).unsqueeze(2).expand_as(yy).transpose(1,2)


    # Compute squared Euclidean distances
    dxx = rx.transpose(1, 2) + rx - 2. * xx
    dyy = ry.transpose(1, 2) + ry - 2. * yy
    dxy = rx.transpose(1, 2) + ry - 2. * zz
    
    # Initialize tensors for results
    XX = torch.zeros_like(xx)
    YY = torch.zeros_like(xx)
    XY = torch.zeros_like(xx)

    if kernel == "multiscale":
        bandwidth_range = [0.01, 0.05, 0.1, 0.2, 0.5]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":
        bandwidth_range = [0.01,0.05,0.1,0.2,0.5,0.7,1.0]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY, [1,2])

def BMMD_Naive(x,y,kernel):
    """
    Naive implementation of BMMD
    """
    lst = []
    for i in tqdm(range(x.shape[1])):
        lst.append(MMD(x[:,i].to(device), y[:,i].to(device), kernel))
        # Call torch.cuda.empty_cache() to release GPU memory
        torch.cuda.empty_cache()
    return torch.tensor(lst).float()

def VDS_Naive(x,y,kernel):
    lst = []
    for i in tqdm(range(x.shape[-1])):
        idx = np.random.randint(0, x.shape[0]*x.shape[1], 10000)
        lst.append(MMD(x[:,:,i].flatten()[idx].unsqueeze(-1).cuda(),y[:,:,i].flatten()[idx].unsqueeze(-1).cuda(), kernel))
        torch.cuda.empty_cache()
    return torch.tensor(lst).float()