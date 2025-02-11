import segmentation_models_pytorch as smp

model_vgg_pt = smp.Unet(
    encoder_name="vgg16",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
    activation='sigmoid'
)
print(model_vgg_pt)
