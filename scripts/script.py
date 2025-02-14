from scripts.image_processor import ImageProcessor

from scripts.image_processor import ImageProcessor
from scripts.unet import Unet

if __name__ == "__main__":
    # Step 1 : Image processing
    processor = ImageProcessor(delete_black_images=True)
    processor.process_files()

    # Step 2 : Chossing model and specify model path
    model_name = input("Enter the model encoder (resnet50 or vgg16): ")
    model_path = input("Enter the path to the pre-downloaded model file (unet_resnet50.pth or unet_vgg16.pth): ")

    # Step 3 : Load and train model
    model = Unet(model_name=model_name, model_path=model_path, dataset_dir="./Data")
    model.train(num_epochs=20)

    # Step 4 : Evaluate model
    model.evaluate_model()
