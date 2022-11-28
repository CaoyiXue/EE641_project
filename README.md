# EE641_project
#### 1. Create Segmentation model with encoder MicroNet


```python
import seg_models as sm

model = sm.Unet(
    encoder_name="micronet_m0",        # choose encoder, e.g. micronet_m0, micronet_m1
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)
```
