import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet152_Weights

from .commons import PositionalEncoding1D, FeatureExtractor

from typing import Dict
from torch import Tensor


class ActionSpotter(nn.Module):

    def __init__(self, input_size=2048, window_size_sec=15, frame_rate=2, num_classes=17, vocab_size=64, weights=None,
                 device=None):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(ActionSpotter, self).__init__()

        self.frames_per_window = window_size_sec * frame_rate
        self.input_size = input_size
        self.num_classes = num_classes
        self.frame_rate = frame_rate
        self.vlad_k = vocab_size

        # are feature already PCA'ed?
        if self.input_size != 256:
            self.feature_extractor = nn.Linear(self.input_size, 256)
            self.input_size = 256

        self.position_embedding = PositionalEncoding1D(self.input_size)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.input_size, nhead=8),
                                             num_layers=2,
                                             norm=nn.LayerNorm(self.input_size))
        self.fc = nn.Linear(self.input_size, self.num_classes + 1)

        self.drop = nn.Dropout(p=0.1)
        self.sigmoid = nn.Sigmoid()

        self.load_weights(weights=weights, device=device)

    def load_weights(self, weights=None, device=None):
        if weights:
            print(f"=> loading checkpoint '{weights}'")
            checkpoint = torch.load(weights, map_location=device)
            self.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{weights}' (epoch {checkpoint['epoch']})")

    def forward(self, x):
        # input_shape: (batch,frames,dim_features)

        batch_size, frames_per_window_, ndim = x.shape
        if ndim != self.input_size:
            x = x.reshape(batch_size * frames_per_window_, ndim)
            x = self.feature_extractor(x)
            x = x.reshape(batch_size, frames_per_window_, -1)

        # Add positional encodings to the input features
        pos_encodings = self.position_embedding(x)
        x = torch.add(x, pos_encodings)
        # Split the feature vector into two non-overlapped subchunks
        half_frames = x.shape[1] // 2
        x1 = self.encoder(x[:, :half_frames, :])
        x2 = self.encoder(x[:, half_frames:, :])
        x = torch.cat((x1, x2), dim=1)
        # Average all frames features within the sequence
        x = torch.mean(x, dim=1)
        # Apply sigmoid and dropout to avoid overfitting
        x = self.sigmoid(x)

        x = self.drop(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


class ActionSpottingPipeline(nn.Module):

    def __init__(self, backbone: str = 'ResNet152', input_size=2048, window_size_sec=15, frame_rate=2, num_classes=17,
                 vocab_size=64, weights=None, device=None):
        super().__init__()

        if backbone == 'ResNet152':
            backbone = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
            backbone.eval()
            self.backbone = FeatureExtractor(backbone, ['avgpool'])

        self.action_spotter = ActionSpotter(input_size, window_size_sec, frame_rate, num_classes, vocab_size,
                                            weights, device)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.backbone(x).squeeze()
        x = x[None, :, :]
        return self.action_spotter(x)
