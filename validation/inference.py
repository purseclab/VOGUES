import torch
import argparse
from model import LSTMAutoencoder
from data import PoseDataset
from torch.utils.data import DataLoader

def validate_sequences(model_path, test_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Explicitly set weights_only=False for safe unpickling of NumPy scalars in PyTorch 2.6+
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    input_dim = checkpoint['input_dim']
    hidden_dim = checkpoint['hidden_dim']
    threshold = checkpoint['threshold']
    
    model = LSTMAutoencoder(input_dim, hidden_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dataset = PoseDataset(test_file_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"Detected {len(dataset)} tracking sequence(s). Validating against threshold: {threshold:.6f}\n")
    
    for idx, (seq_tensor, lengths) in enumerate(dataloader):
        seq_tensor = seq_tensor.to(device)
        
        with torch.no_grad():
            reconstructed = model(seq_tensor, lengths)
            criterion = torch.nn.MSELoss(reduction='none')
            loss_tensor = criterion(reconstructed, seq_tensor)
            
            seq_len = lengths[0].item()
            loss = loss_tensor[0, :seq_len, :].mean().item()
            
        is_valid = loss <= threshold
        status = "valid" if is_valid else "invalid"
        
        print(f"--- Sequence {idx + 1} ---")
        print(f"Result: {status.upper()}")
        print(f"Reconstruction Loss: {loss:.6f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='one_class_lstm.pth')
    parser.add_argument('--test_sequence_path', type=str, required=True, help='Path to evaluation JSON')
    args = parser.parse_args()
    
    validate_sequences(args.model_path, args.test_sequence_path)