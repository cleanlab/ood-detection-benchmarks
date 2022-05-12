import pytorch_lightning as pl
import torchvision.models as models
from torch import nn


class LitFeatureExtractor(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(pretrained=True)
        self.num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, X):
        features = self.feature_extractor(X).flatten(1)
        return features
