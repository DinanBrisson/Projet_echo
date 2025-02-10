import segmentation_models_pytorch as smp

model_resnet_pt = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=1,
    classes=2,
    activation='softmax'
)
print(model_resnet_pt)
