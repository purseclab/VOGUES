import torch
import numpy as np

def calculate_reconstruction_threshold(model, dataloader, device):
    model.eval()
    losses = []
    criterion = torch.nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for batch, lengths in dataloader:
            batch = batch.to(device)
            pred = model(batch, lengths)
            loss = criterion(pred, batch)
            
            for i, length in enumerate(lengths):
                seq_loss = loss[i, :length, :].mean().item()
                losses.append(seq_loss)
                
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    
    if std_loss == 0:
        threshold = mean_loss * 1.15
    else:
        threshold = mean_loss + (2.0 * std_loss)
        
    return threshold, mean_loss