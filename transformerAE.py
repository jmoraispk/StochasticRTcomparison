import torch
import torch.nn as nn
from typing import Optional

class Encoder(nn.Module):
    # input: (batch_size, Nc, Nt) channel matrix
    # output: (batch_size, encoded_dim) codeword
    # CSI_NET
    def __init__(self, encoded_dim, Nc=100, Nrt=64, n_heads=8):
        super().__init__()
        self.Nc = Nc
        self.Nrt = Nrt
        self.n_out_channels = 8
        self.encoded_dim = encoded_dim

        self.conv_block1 = nn.Conv2d(in_channels=2, 
                      out_channels=self.n_out_channels, 
                      kernel_size=(1,1), # (3,3)
                      stride=1, 
                      padding=0, 
                      bias=True)
        
        self.d_model = self.Nrt * self.n_out_channels
        dim_feedforward = 4 * self.d_model
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, 
                                                  dim_feedforward=dim_feedforward, 
                                                  nhead=n_heads, 
                                                  batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Update input features to account for all dimensions
        self.fc = nn.Linear(in_features=self.d_model * self.Nc, out_features=encoded_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.name = "TransformerAE"

    def forward(self, x):
        # x.shape = (batch_size, 2, Nc, Nt)
        out = self.conv_block1(x)
        # out.shape = (batch_size, n_out_channels, Nc, Nt)

        # Reshape to (batch_size, Nc, d_model)
        out = out.permute(0, 2, 1, 3)  # (batch_size, Nc, n_out_channels, Nt)
        out = out.reshape(out.shape[0], self.Nc, self.d_model)
        # out.shape = (batch_size, Nc, d_model)
        
        out = self.transformer(out)
        # out.shape = (batch_size, Nc, d_model)

        # Flatten all dimensions except batch
        out = out.reshape(out.shape[0], -1)  # (batch_size, Nc * d_model)
        out = self.sigmoid(self.fc(out))  # (batch_size, encoded_dim)
        # out.shape = (batch_size, encoded_dim)
        
        return out

class Decoder(nn.Module):
    # input: (batch_size, encoded_dim) codeword
    # output: (batch_size, 2, Nc, Nt) channel matrix
    def __init__(self, encoded_dim, Nc=100, Nrt=64):
        super().__init__()
        self.name = "TransformerAE-Decoder"
        self.Nc = Nc
        self.Nrt = Nrt
        self.n_out_channels = 8
        self.d_model = self.Nrt * self.n_out_channels

        self.fc = nn.Linear(in_features=encoded_dim, out_features=self.d_model)
        
        dim_feedforward = 4 * self.d_model
        nhead = 8
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, 
                                                  dim_feedforward=dim_feedforward, 
                                                  nhead=nhead, 
                                                  batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.conv_block1 = nn.Conv2d(in_channels=self.n_out_channels, 
                      out_channels=2, 
                      kernel_size=(1,1), # (3,3)
                      stride=1, 
                      padding=0, 
                      bias=True)

    def forward(self, x):
        # x.shape = (batch_size, encoded_dim)
        out = self.fc(x)  # (batch_size, d_model)
        # out.shape = (batch_size, d_model)

        # Expand to sequence length Nc
        out = out.unsqueeze(1).expand(-1, self.Nc, -1)  # (batch_size, Nc, d_model)
        # out.shape = (batch_size, Nc, d_model)
        
        out = self.transformer(out)
        # out.shape = (batch_size, Nc, d_model)

        # Reshape for conv layer
        out = out.reshape(out.shape[0], self.Nc, self.n_out_channels, self.Nrt)
        # out.shape = (batch_size, Nc, n_out_channels, Nrt)
        out = out.permute(0, 2, 1, 3)  # (batch_size, n_out_channels, Nc, Nrt)
        # out.shape = (batch_size, n_out_channels, Nc, Nrt)
        out = self.conv_block1(out)
        # out.shape = (batch_size, 2, Nc, Nrt)
        
        return out
    
# def quantize(x, k):
#     n = float(2 ** k - 1)

#     @tf.custom_gradient
#     def _quantize(x):
#         return tf.round(x * n) / n, lambda grad: grad

#     return _quantize(x)

# def quantize_layer(x, k_bits):
#     q_encoded = Lambda(quantize, arguments={'k': k_bits}, name='quantize')(x)
#     return q_encoded

class QuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2 ** k - 1)
        return torch.round(input * n) / n

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # No gradient for `k`

def quantize(x, k):
    return QuantizeFunction.apply(x, k)

class QuantizeLayer(nn.Module):
    def __init__(self, k_bits):
        super().__init__()
        self.k_bits = k_bits

    def forward(self, x):
        return quantize(x, self.k_bits)

class TransformerAE(nn.Module):
    """
    TransformerAE

    k_bits: Optional[int] = None
    """
    def __init__(self, encoded_dim, Nc=100, Nt=64, k_bits: Optional[int] = None):
        super().__init__()
        self.encoder = Encoder(encoded_dim, Nc, Nt)
        self.quantize = bool(k_bits)
        self.k_bits = k_bits
        if self.quantize:
            self.quantize_layer = QuantizeLayer(k_bits=encoded_dim*2)

        self.decoder = Decoder(encoded_dim, Nc, Nt)
        self.name = self.encoder.name + '-' + self.decoder.name
    
    def forward(self, x):
        # x.shape = (batch_size, 2, Nc, Nt)
        # print(f'x.shape: {x.shape}')
        encoded_vector = self.encoder(x)
        if self.quantize:
            encoded_vector = self.quantize_layer(encoded_vector)
            # encoded_vector = quantize(encoded_vector, self.k_bits)
            
        x_recovered = self.decoder(encoded_vector)
        # print(f'x_recovered.shape: {x_recovered.shape}')
        
        return encoded_vector, x_recovered
    