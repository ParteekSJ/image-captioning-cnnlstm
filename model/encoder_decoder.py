from model.encoder import CNNModel
from model.decoder import LSTMModel
from torch import nn


class CNNLSTMModel(nn.Module):
    def __init__(self, encoder: CNNModel, decoder: LSTMModel):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, capts, lengths):
        input_features = self.encoder(images)
        output = self.decoder(input_features, capts, lengths)
        return output
