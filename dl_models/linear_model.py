import math
import torch
from torch import nn
from torch.nn import functional as F


class BasicLinear(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True):
        super(BasicLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return {'logits': F.linear(input, self.weight, self.bias)}


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}

class TopKCosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, sigma=True):
        super(TopKCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy  # Total prototypes
        self.nb_proxy = nb_proxy
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, X, topk_indices=None, class_task_query=None):
        if topk_indices == None:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

            if self.sigma is not None:
                out = self.sigma * out

            return {'logits': out}
        else:

            T, N, D = X.shape
            K = topk_indices.size(1)
            with torch.no_grad():
                task_indices = class_task_query(topk_indices)  # [N,K]

            norm_weight = F.normalize(self.weight,dim=-1)

            sample_idx = torch.arange(N, device=X.device)[:, None].expand(N, K)  # [N,K]
            task_idx = task_indices.long()  # [N,K]
            
            task_features = X[task_idx, sample_idx, :]  # [N,K,D]
            
            selected_weights = norm_weight[topk_indices]
            
            similarities = torch.einsum('nkd,nkd->nk', 
                                    F.normalize(task_features, dim=-1),
                                    selected_weights)
            index = similarities.argmax(1)
            
            label = topk_indices.gather(1, index.unsqueeze(1)).squeeze(1)
            #print(index,label)
            #print(topk_indices)
            return label
        
class SecondCosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(SecondCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input, class_index=None, use_kl_div = True):
        if class_index == None:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else: 
            if use_kl_div:
                out = KL_div(input,self.weight[class_index])
            else:
                out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight[class_index], p=2, dim=1))
       
        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out
        
        return {'logits': out}


class RandomProjectLinear(nn.Module):
    def __init__(self, in_features, M=10000 ):
        super(RandomProjectLinear, self).__init__()
        self.project = nn.Linear(in_features,M)
        self.reproject= nn.Linear(M,in_features)
        self.init_weight()
    
    def init_weight(self):
        stdv1 = 1. / math.sqrt(self.project.weight.size(1))
        stdv2 = 1. / math.sqrt(self.reproject.weight.size(1))

        self.project.weight.data.uniform_(-stdv1, stdv1)
        self.reproject.weight.data.uniform_(-stdv2, stdv2) 
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, input):
        out = self.reproject(F.relu(self.project(input)))
        
        return out

def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)


def KL_div(_p, _q):

    return  torch.sum(_p * (_p.log() - _q.log()), dim=-1)