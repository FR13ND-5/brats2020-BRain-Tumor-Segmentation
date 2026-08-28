"""Model registry — adding a model = one file + one entry here."""
from .unet import UNet
from .attention_unet import AttentionUNet
from .unetpp import UNetPP
from .transunet import TransUNet

MODELS = {
    "unet": UNet,
    "attention_unet": AttentionUNet,
    "unetpp": UNetPP,
    "transunet": TransUNet,
}


def get_model(name, in_channels=4, out_channels=3):
    return MODELS[name](in_channels=in_channels, out_channels=out_channels)
