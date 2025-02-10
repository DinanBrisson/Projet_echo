import segmentation_models_pytorch as smp

model_vgg_pt = smp.Unet(
    encoder_name="vgg16",
    encoder_weights="imagenet",
    in_channels=1,
    classes=2,
    activation='softmax'
)
print(model_vgg_pt)
