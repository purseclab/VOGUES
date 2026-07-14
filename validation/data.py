import os
import json
import torch
from torch.utils.data import Dataset

class PoseDataset(Dataset):
    def __init__(self, json_path, max_len=None):
        # Load the raw JSON file
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        
        self.sequences = []
        self.lengths = []
        
        for seq in raw_data:
            sorted_seq = sorted(seq, key=lambda x: x[0])
            kp_seq = [frame[1] for frame in sorted_seq]
            
            self.sequences.append(torch.tensor(kp_seq, dtype=torch.float32))
            self.lengths.append(len(kp_seq))
        
        self.max_len = max_len if max_len else max(self.lengths)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        length = self.lengths[idx]
        
        pad_size = self.max_len - length
        if pad_size > 0:
            padding = torch.zeros((pad_size, seq.shape[1]), dtype=torch.float32)
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
            
        return padded_seq, length