# EE641_project
# Reference: [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
#### 1. Create your first Segmentation model with SMP

Segmentation model is just a PyTorch nn.Module, which can be created as easy as:

```python
import seg_models as sm

model = sm.Unet(
    encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
)
```
