import os
import json
import torch
import re
from torch.utils.data import Dataset

class PoseDataset(Dataset):
    def __init__(self, json_path, max_len=None):
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        
        self.sequences = []
        self.lengths = []
        
        if len(raw_data) == 0:
            raise ValueError("JSON file is empty.")
            
        if isinstance(raw_data[0], dict) and "image_id" in raw_data[0]:
            self._parse_alphapose(raw_data)
        elif isinstance(raw_data[0], list):
            self._parse_custom(raw_data)
        else:
            raise ValueError("Unrecognized JSON format. Must be pose dicts or custom nested lists.")
            
        self.max_len = max_len if max_len else max(self.lengths)
        
        for i in range(len(self.sequences)):
            seq = self.sequences[i]
            seq_min = seq.min(dim=0, keepdim=True)[0]
            seq_max = seq.max(dim=0, keepdim=True)[0]
            denom = seq_max - seq_min
            denom[denom == 0] = 1e-6 
            self.sequences[i] = (seq - seq_min) / denom
        
    def _parse_custom(self, raw_data):
        for seq in raw_data:
            sorted_seq = sorted(seq, key=lambda x: x[0])
            kp_seq = [frame[1] for frame in sorted_seq]
            self.sequences.append(torch.tensor(kp_seq, dtype=torch.float32))
            self.lengths.append(len(kp_seq))

    def _parse_alphapose(self, raw_data):
        def extract_frame_num(image_id):
            numbers = re.findall(r'\d+', image_id)
            return int(numbers[-1]) if numbers else image_id
            
        sorted_data = sorted(raw_data, key=lambda x: extract_frame_num(x["image_id"]))
        
        grouped_seqs = {}
        for item in sorted_data:
            if item.get("category_id", 1) == 1:
                seq_id = item.get("sequence_id", item.get("idx", 1))
                if seq_id not in grouped_seqs:
                    grouped_seqs[seq_id] = []
                grouped_seqs[seq_id].append(item["keypoints"])
                
        for seq_id, kp_seq in grouped_seqs.items():
            self.sequences.append(torch.tensor(kp_seq, dtype=torch.float32))
            self.lengths.append(len(kp_seq))

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