import torch
import torch.nn.functional as F

def asymmetric_focal_loss(y_true, y_pred, delta, gamma, epsilon=1e-7):
    y_true = F.one_hot(y_true, 2)
    y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * torch.log(y_pred)
        
    back_ce = torch.pow(1 - y_pred[:,0], gamma) * cross_entropy[:,0]
    back_ce =  (1 - delta) * back_ce

    fore_ce = cross_entropy[:,1] # num_points, num_features
    fore_ce = delta * fore_ce

    loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))
    return loss

def asymmetric_focal_tversky_loss(y_true, y_pred, delta, gamma, epsilon=1e-7):
    mask = y_true.to(torch.bool)
    y_true_neg = (1-y_true)[~mask]
    mTI_0 = (torch.sum(y_true_neg * y_pred[~mask, 0]) + epsilon) / ((torch.sum(y_true_neg * y_pred[~mask, 0]) + delta * torch.sum(y_true[mask] * y_pred[mask, 0]) + (1 - delta) * torch.sum(y_true_neg * y_pred[~mask, 1])) + epsilon)
    mTI_1 = (torch.sum(y_true[mask] * y_pred[mask, 1]) + epsilon) / ((torch.sum(y_true[mask] * y_pred[mask, 1]) + (1-delta) * torch.sum(y_true_neg * y_pred[~mask, 1]) + (delta) * torch.sum(y_true[mask] * y_pred[mask, 0])) + epsilon)
    
    loss = 1.0 - mTI_0 + torch.pow(1.0 - mTI_1, 1.0 - gamma)
    return loss

def asym_unified_focal_loss(y_true, y_pred, weight=0.999, 
                            delta_focal=0.6, gamma_focal=0.5,
                            delta_tversky=0.5, gamma_tversky=1):
    # need to tune gamma [0.1, 0.9]
    focal_loss = asymmetric_focal_loss(y_true, y_pred, delta_focal, gamma_focal)
    tversky_loss = asymmetric_focal_tversky_loss(y_true, y_pred, delta_tversky, gamma_tversky)

    #print("FOCAL w", weight * focal_loss)
    #print("Focal", focal_loss)
    #print("Tversky w", (1 - weight) * tversky_loss)
    #print("Tversky", tversky_loss)
    loss = 100 * (focal_loss * weight + tversky_loss * (1 - weight))
    return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, assymetric=False, weight=1, 
                 delta_focal=0.5, gamma_focal=1, 
                 delta_tversky=0.5, gamma_tversky=1):
        super(FocalLoss,self).__init__()
        self.assymetric = assymetric
        # focal: delta=0.999, gamma=0.2 tversky_loss: delta=0.9, gamma=0.5
        self.weight = weight
        self.delta_focal = delta_focal
        self.gamma_focal = gamma_focal
        self.delta_tversky = delta_tversky
        self.gamma_tversky = gamma_tversky

    def forward(self, pred, true):
        if self.assymetric:
            return asym_unified_focal_loss(true, pred.softmax(dim=1), self.weight, 
                    self.delta_focal, self.gamma_focal,
                    self.delta_tversky, self.gamma_tversky
                    )
        else:
            raise NotImplemented

