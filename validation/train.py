import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import PoseDataset
from model import LSTMAutoencoder
import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to JSON of valid pose data')
    parser.add_argument('--model_save_path', type=str, default='one_class_lstm.pth')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = PoseDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    sample_seq, _ = dataset[0]
    input_dim = sample_seq.shape[1]
    
    model = LSTMAutoencoder(input_dim, args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss(reduction='none')
    
    print(f"Training One-Class Autoencoder on {len(dataset)} sequences (Feature Dim: {input_dim}).")
    
    model.train()
    for epoch in range(args.num_epochs):
        total_loss = 0
        for batch, lengths in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred = model(batch, lengths)
            loss = criterion(pred, batch)
            
            mask = torch.zeros_like(loss)
            for i, length in enumerate(lengths):
                mask[i, :length, :] = 1.0
                
            masked_loss = (loss * mask).sum() / mask.sum()
            masked_loss.backward()
            optimizer.step()
            
            total_loss += masked_loss.item()
            
        print(f"Epoch {epoch+1}/{args.num_epochs} | Training Loss: {total_loss/len(dataloader):.6f}")
        
    threshold, mean_loss = utils.calculate_reconstruction_threshold(model, dataloader, device)
    print(f"\n--- Training Complete ---")
    print(f"Normal Pose Mean Loss: {mean_loss:.6f}")
    print(f"Computed Valid Threshold: {threshold:.6f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': threshold,
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim
    }, args.model_save_path)
    print(f"Model saved to {args.model_save_path}")

if __name__ == '__main__':
    main()