import torch
import torch.nn as nn
import torchvision.models as models
from timm.models import register_model
import logging

logging.basicConfig(
    format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('./resPhosNetLogs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('resphoscnet')
logger.info('--- resphoscnet ---')

__all__ = ["ResNet51PretrainOctopus"]

class TemporalPyramidPooling(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.levels = levels

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        pooled_features = []
        for level in self.levels:
            bin_width = width // level
            for i in range(level):
                start = i * bin_width
                end = start + bin_width if i < level - 1 else width
                if end > start:
                    pooled = torch.max_pool2d(x[:, :, :, start:end], (height, end - start))
                    pooled_features.append(pooled.view(batch_size, channels))
        return torch.cat(pooled_features, dim=1)

class ResNet34Pretrain(nn.Module):
    def __init__(self, phos_size=180, phoc_size=646):
        super().__init__()
        
        base_model = models.resnet34(pretrained=True)
        logger.info("LOADED RESNET34 PRETRAINED MODEL!!!")

        self.conv = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4  # Output: (B, 512, H, W)
        )

        # PHOC TPP layers for each pyramidal level
        self.phoc_tpp_2 = TemporalPyramidPooling([2])  # For 2 splits (78)
        self.phoc_tpp_3 = TemporalPyramidPooling([3])  # For 3 splits (117)
        self.phoc_tpp_4 = TemporalPyramidPooling([4])  # For 4 splits (156)
        self.phoc_tpp_5 = TemporalPyramidPooling([5])  # For 5 splits (195)
        self.phoc_tpp_bigram = TemporalPyramidPooling([2])  # For bigrams (100)

        # PHOC linear layers for each level
        self.phoc_2 = nn.Sequential(
            nn.Linear(512 * 2, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 72),  # 2 splits × 39
            nn.Sigmoid()
        )
        self.phoc_3 = nn.Sequential(
            nn.Linear(512 * 3, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 108),  # 3 splits × 39
            nn.Sigmoid()
        )
        self.phoc_4 = nn.Sequential(
            nn.Linear(512 * 4, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 144),  # 4 splits × 39
            nn.Sigmoid()
        )
        self.phoc_5 = nn.Sequential(
            nn.Linear(512 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 180),  # 5 splits × 39
            nn.Sigmoid()
        )
        self.phoc_bigram = nn.Sequential(
            nn.Linear(512 * 2, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 100),  # 2 halves × 50
            nn.Sigmoid()
        )

        # PHOS single TPP and linear layers
        self.phos_tpp = TemporalPyramidPooling([1, 2, 5])  # 1+2+5=8 bins
        self.phos = nn.Sequential(
            nn.Linear(512 * 8, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, phos_size),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.conv(x)  # (B, 512, H, W)

        # PHOC branch: separate TPP and linear layers for each level
        phoc_2 = self.phoc_tpp_2(x)  # (B, 512*2)
        phoc_3 = self.phoc_tpp_3(x)  # (B, 512*3)
        phoc_4 = self.phoc_tpp_4(x)  # (B, 512*4)
        phoc_5 = self.phoc_tpp_5(x)  # (B, 512*5)
        phoc_bigram = self.phoc_tpp_bigram(x)  # (B, 512*2)

        #print("0.",phoc_2.shape,phoc_3.shape,phoc_4.shape,phoc_5.shape,phoc_bigram.shape)

        phoc_2_out = self.phoc_2(phoc_2)  # (B, 78)
        phoc_3_out = self.phoc_3(phoc_3)  # (B, 117)
        phoc_4_out = self.phoc_4(phoc_4)  # (B, 156)
        phoc_5_out = self.phoc_5(phoc_5)  # (B, 195)
        phoc_bigram_out = self.phoc_bigram(phoc_bigram)  # (B, 100)

        #print("2.",phoc_2_out.shape,phoc_3_out.shape,phoc_4_out.shape,phoc_5_out.shape,phoc_bigram_out.shape)


        # Concatenate PHOC outputs
        phoc_out = torch.cat([phoc_2_out, phoc_3_out, phoc_4_out, phoc_5_out, phoc_bigram_out], dim=1)  # (B, 646)

        # PHOS branch
        phos_out = self.phos_tpp(x)  # (B, 512*8)
        phos_out = self.phos(phos_out)  # (B, 180)

        return {'phos': phos_out, 'phoc': phoc_out}

    def preload_conv_layer(self, weights_file):
        self.conv.load_state_dict(torch.load(weights_file))

@register_model
def ResNet51PretrainOctopus(**kwargs):
    return ResNet34Pretrain(
        phos_size=kwargs.get('phos_size', 165),
        phoc_size=kwargs.get('phoc_size', 604),
    )

if __name__ == "__main__":
    model = ResNet34Pretrain(
        phos_size=165,
        phoc_size=604,
    )

    x = torch.rand([5, 3, 64, 256])
    out = model(x)
    print("\n\t out.shape:", out["phoc"].shape, out["phos"].shape)