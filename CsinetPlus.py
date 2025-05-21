import torch
import torch.nn as nn
# from torchinfo import summary

Nc = 32  # The number of subcarriers
Nt = 32  # The number of transmit antennas


NC = 100  # The number of subcarriers
NT = 64  # The number of transmit antennas

class Encoder(nn.Module):
    # input: (batch_size, Nc, Nt) channel matrix
    # output: (batch_size, encoded_dim) codeword
    # CSI_NET
    def __init__(self, encoded_dim, Nc=NC, Nt=NT):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(num_features=2),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(num_features=2),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.fc = nn.Linear(in_features=2 * Nc * Nt, out_features=encoded_dim)
        self.leakyrelu = nn.LeakyReLU()
        self.name = "CsinetPlus"

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        # out.shape = (batch_size, 2, Nc, Nt)
        # if test: show(out)
        out = torch.reshape(out, (out.shape[0], -1))
        # out.shape = (batch_size, 2*Nc*Nt)
        out = self.fc(out)
        # if test: show(torch.reshape(out, (batch_size, 1, 4, encoded_dim//4)))
        # out.shape = (batch_size, encoded_dim)

        return out


class Refinenet(nn.Module):
    # input: (batch_size, 2, Nc, Nt)
    # output: (batch_size, 2, Nc, Nt)
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=2),
            nn.Tanh(),
        )

    def forward(self, x):
        skip_connection = x
        out = self.conv1(x)
        # out.shape = (batch_size, 8, Nc, Nt)
        out = self.conv2(out)
        # out.shape = (batch_size, 16, Nc, Nt)
        out = self.conv3(out)
        # out.shape = (batch_size, 2, Nc, Nt)
        out = out + skip_connection

        return out
    

class Decoder(nn.Module):
    # input: (batch_size, encoded_dim) codeword
    # output: (batch_size, Nc, Nt) reconstructed channel matrix
    # CSI_NET
    def __init__(self, encoded_dim, Nc=NC, Nt=NT, n_refine_nets=5):
        super().__init__()
        self.fc = nn.Linear(in_features=encoded_dim, out_features=2 * Nc * Nt)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(num_features=2),
            nn.Tanh(),
        )
        self.Nc = Nc
        self.Nt = Nt
        self.n_refine_nets = n_refine_nets
        # Why doesn't this loop work?
        # self.refine_layer_list = []
        # for layer_idx in range(n_refine_nets):
        #     self.refine_layer_list.append(Refinenet())
        
        if self.n_refine_nets >= 1:
            self.refine1 = Refinenet()
        if self.n_refine_nets >= 2:    
            self.refine2 = Refinenet()
        if self.n_refine_nets >= 3:
            self.refine3 = Refinenet()
        if self.n_refine_nets >= 4:
            self.refine4 = Refinenet()
        if self.n_refine_nets >= 5:
            self.refine5 = Refinenet()
        if self.n_refine_nets >= 6:
            self.refine6 = Refinenet()
        if self.n_refine_nets >= 7:
            self.refine7 = Refinenet()
        if self.n_refine_nets >= 8:
            self.refine8 = Refinenet()
            
        self.name = "CsinetPlus"


    def forward(self, x):
        # x.shape = (batch_size, encoded_dim)
        out = self.fc(x)
        # out.shape = (batch_size, 2*Nc*Nt)
        out = torch.reshape(out, (out.shape[0], 2, self.Nc, self.Nt))
        # out.shape = (batch_size, 2, Nc, Nt)
        out = self.conv_block1(out)
        
        # Not sure why this doesn't work
        # for refine_layer in self.refine_layer_list:
        #     out = refine_layer(out)
        
        if self.n_refine_nets >= 1:    
            out = self.refine1(out)
        if self.n_refine_nets >= 2:    
            out = self.refine2(out)
        if self.n_refine_nets >= 3:
            out = self.refine3(out)
        if self.n_refine_nets >= 4:
            out = self.refine4(out)
        if self.n_refine_nets >= 5:
            out = self.refine5(out)
        if self.n_refine_nets >= 6:
            out = self.refine6(out)
        if self.n_refine_nets >= 7:
            out = self.refine7(out)
        if self.n_refine_nets >= 8:
            out = self.refine8(out)
            
        tmp = out.reshape((out.shape[0], -1))
        tmp = torch.norm(tmp,dim=(-1), keepdim=True)
        tmp = tmp.reshape((tmp.shape[0], 1, 1, 1))
        out = out / tmp

        return out
    


class CsinetPlus(nn.Module):
    def __init__(self, encoded_dim, Nc=NC, Nt=NT, n_refine_nets=5):
        super().__init__()
        self.encoder = Encoder(encoded_dim, Nc, Nt)
        self.decoder = Decoder(encoded_dim, Nc, Nt, n_refine_nets)
        self.name = self.encoder.name + '-' + self.decoder.name
    
    def forward(self, x):
        encoded_vector = self.encoder(x)
        x_recovered = self.decoder(encoded_vector)

        return encoded_vector, x_recovered
    

if __name__ == "__main__":
    encoded_dim = 32
    encoder = Encoder(encoded_dim)
    decoder = Decoder(encoded_dim)
    autoencoder = CsinetPlus(encoded_dim)
    # summary(encoder, input_size=(16, 2, NC, NT))
    # summary(decoder, input_size=(16, encoded_dim))
    # summary(autoencoder, input_size=(16, 2, NC, NT))
    
    print("done")
