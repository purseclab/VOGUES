import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, lengths):
        batch_size, seq_len, input_dim = x.size()
        
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, cn) = self.encoder(packed_input)
        
        latent = hn[-1].unsqueeze(1) 
        decoder_input = latent.repeat(1, seq_len, 1)
        
        dec_out, _ = self.decoder(decoder_input)
        reconstructed = self.output_layer(dec_out)
        return reconstructed