import keras
import segmentation_models as sm

sm.set_framework('tf.keras')
sm.framework()

# U-Net ResNet50
model_resnet = sm.Unet(
    backbone_name='resnet50',
    encoder_weights='imagenet',
    input_shape=(256, 256),
    classes=2,
    activation='softmax'
)
model_resnet.summary()

