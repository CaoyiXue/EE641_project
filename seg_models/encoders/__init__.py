import torch.utils.model_zoo as model_zoo

from .resnet import resnet_encoders
from .mobilenet import mobilenet_encoders
from .micronet import micronet_encoders
encoders = {}
encoders.update(resnet_encoders)
encoders.update(mobilenet_encoders)
encoders.update(micronet_encoders)

def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError(
                "Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights,
                    name,
                    list(encoders[name]["pretrained_settings"].keys()),
                )
            )
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels, pretrained=weights is not None)
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder