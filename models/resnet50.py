import segmentation_models_pytorch as smp

model_resnet_pt = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
    activation='sigmoid'
)
print(model_resnet_pt)
