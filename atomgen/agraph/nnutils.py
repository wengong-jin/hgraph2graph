import torch
import torch.nn as nn
import torch.nn.functional as F

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

def avg_pool(all_vecs, scope, dim):
    size = create_var(torch.Tensor([le for _,le in scope]))
    return all_vecs.sum(dim=dim) / size.unsqueeze(-1)

def get_accuracy_bin(scores, labels):
    preds = torch.ge(scores, 0).long()
    acc = torch.eq(preds, labels).float()
    return torch.sum(acc) / labels.nelement()

def get_accuracy(scores, labels):
    _,preds = torch.max(scores, dim=-1)
    acc = torch.eq(preds, labels).float()
    return torch.sum(acc) / labels.nelement()

def get_accuracy_sym(scores, labels):
    max_scores,max_idx = torch.max(scores, dim=-1)
    lab_scores = scores[torch.arange(len(scores)), labels]
    acc = torch.eq(lab_scores, max_scores).float()
    return torch.sum(acc) / labels.nelement()

def stack_pad_tensor(tensor_list):
    max_len = max([t.size(0) for t in tensor_list])
    for i,tensor in enumerate(tensor_list):
        pad_len = max_len - tensor.size(0)
        tensor_list[i] = F.pad( tensor, (0,0,0,pad_len) )
    return torch.stack(tensor_list, dim=0)

def create_pad_tensor(alist):
    max_len = max([len(a) for a in alist]) + 1
    for a in alist:
        pad_len = max_len - len(a)
        a.extend([0] * pad_len)
    return torch.IntTensor(alist)

def zip_tensors(tup_list):
    arr0, arr1, arr2 = list(zip(*tup_list))
    if type(arr2[0]) is int: 
        arr0 = torch.stack(arr0, dim=0)
        arr1 = torch.LongTensor(arr1).cuda() 
        arr2 = torch.LongTensor(arr2).cuda() 
    else:
        arr0 = torch.cat(arr0, dim=0)
        arr1 = [x for a in arr1 for x in a]
        arr1 = torch.LongTensor(arr1).cuda() 
        arr2 = torch.cat(arr2, dim=0)
    return arr0, arr1, arr2

def index_scatter(sub_data, all_data, index):
    d0, d1 = all_data.size()
    buf = torch.zeros_like(all_data).scatter_(0, index.repeat(d1, 1).t(), sub_data)
    mask = torch.ones(d0, device=all_data.device).scatter_(0, index, 0)
    return all_data * mask.unsqueeze(-1) + buf

