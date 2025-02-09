import segmentation_models as sm

# Configuration générale
sm.set_framework('tf.keras')
sm.framework()

# U-Net VGG16
model_vgg = sm.Unet(
    backbone_name='vgg16',
    encoder_weights='imagenet',
    input_shape=(256, 256),
    classes=2,
    activation='softmax'
)
model_vgg.summary()
