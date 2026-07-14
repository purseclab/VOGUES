import torch
import argparse
import json
from model import LSTMAutoencoder

def validate_sequence(model_path, raw_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    input_dim = checkpoint['input_dim']
    hidden_dim = checkpoint['hidden_dim']
    threshold = checkpoint['threshold']
    
    model = LSTMAutoencoder(input_dim, hidden_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if isinstance(raw_data[0][0], list):
        sequences = raw_data
        print(f"Detected {len(sequences)} sequences in the file. Validating all...\n")
    else:
        sequences = [raw_data]
        
    for idx, seq in enumerate(sequences):
        sorted_seq = sorted(seq, key=lambda x: x[0])
        kp_seq = [frame[1] for frame in sorted_seq]
        
        seq_tensor = torch.tensor([kp_seq], dtype=torch.float32).to(device)
        lengths = torch.tensor([len(kp_seq)])
        
        with torch.no_grad():
            reconstructed = model(seq_tensor, lengths)
            criterion = torch.nn.MSELoss(reduction='mean')
            loss = criterion(reconstructed, seq_tensor).item()
            
        is_valid = loss <= threshold
        status = "valid" if is_valid else "invalid"
        
        print(f"--- Sequence {idx + 1} ---")
        print(f"Validation Result: {status.upper()}")
        print(f"Reconstruction Loss: {loss:.6f} | Maximum Allowed Threshold: {threshold:.6f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='one_class_lstm.pth')
    parser.add_argument('--test_sequence_path', type=str, required=True, help='Path to JSON sequence file')
    args = parser.parse_args()
    
    with open(args.test_sequence_path, 'r') as f:
        test_data = json.load(f)
        
    validate_sequence(args.model_path, test_data)