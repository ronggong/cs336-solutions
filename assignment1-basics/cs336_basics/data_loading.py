import torch
import numpy as np

def get_batch_at_start(dataset, index, context_length, device):
    seq = []
    next_token = []
    for i in range(len(index)):
        start = index[i]
        seq.append(torch.from_numpy(dataset[start:start+context_length]))
        next_token.append(torch.from_numpy(dataset[start+1:start+context_length+1]))
    return torch.stack(seq, dim=0).to(device=device, dtype=torch.int), torch.stack(next_token, dim=0).to(device=device, dtype=torch.int)

def get_batch(dataset, batch_size, context_length, device):
    len_data = len(dataset)
    assert len_data - 1 >= context_length
    index = np.random.randint(0, len_data - context_length, size=(batch_size,))
    return get_batch_at_start(dataset, index, context_length, device) 
