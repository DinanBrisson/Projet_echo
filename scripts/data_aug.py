import os
import cv2
import numpy as np
import random
import albumentations as A


class DataAugmentation:
    def __init__(self, input_folder, output_folder, num_augmentations):
        """
        Initializes the DataAugmentation class.

        :param input_folder: Path where cropped images and masks are stored.
        :param output_folder: Path where augmented data will be saved.
        :param num_augmentations: Number of augmented versions per original image.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_augmentations = num_augmentations

        self.train_folder = os.path.join(output_folder, "train")
        self.test_folder = os.path.join(output_folder, "test")
        self.val_folder = os.path.join(output_folder, "val")

        # Create necessary directories
        for folder in [self.train_folder, self.test_folder, self.val_folder]:
            os.makedirs(os.path.join(folder, "images"), exist_ok=True)
            os.makedirs(os.path.join(folder, "masks"), exist_ok=True)

        # Define augmentation transforms (only for images)
        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ])

    def augment_and_save(self, img_path, mask_path, save_folder):
        """
        Augments only the cropped image and saves the associated mask with the same name.

        :param img_path: Path to the original cropped grayscale image.
        :param mask_path: Path to the corresponding mask.
        :param save_folder: Folder to save augmented images/masks (train, test, val).
        """
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Save original image and mask to the dataset
        cv2.imwrite(os.path.join(save_folder, "images", f"{base_name}.png"), image)
        cv2.imwrite(os.path.join(save_folder, "masks", f"{base_name}.png"), mask)

        # Perform augmentations on images only
        for i in range(self.num_augmentations):
            augmented = self.augmentation_pipeline(image=image)  # Only augment image
            aug_image = augmented["image"]

            img_save_path = os.path.join(save_folder, "images", f"{base_name}_aug{i}.png")
            mask_save_path = os.path.join(save_folder, "masks", f"{base_name}_aug{i}.png")  # Copy the same mask

            cv2.imwrite(img_save_path, aug_image)
            cv2.imwrite(mask_save_path, mask)  # Save the original mask with a matching augmented filename

            print(f"Augmented image and mask saved: {img_save_path} & {mask_save_path}")

    def process_dataset(self):
        """
        Reads all images and masks, applies augmentation (only to images), and saves them in train/test/val folders.
        """
        images = sorted([f for f in os.listdir(self.input_folder) if f.endswith("_cropped_gray.jpg")])
        masks = sorted([f for f in os.listdir(self.input_folder) if f.endswith("_mask.png")])

        assert len(images) == len(masks), "Mismatch between images and masks"

        # Split dataset (80% Train, 10% Test, 10% Val)
        total_images = len(images)
        train_split = int(0.8 * total_images)
        test_split = int(0.9 * total_images)  # Remaining 10% for validation

        for idx, (img_file, mask_file) in enumerate(zip(images, masks)):
            img_path = os.path.join(self.input_folder, img_file)
            mask_path = os.path.join(self.input_folder, mask_file)

            # Assign to train, test, or validation folder
            if idx < train_split:
                save_folder = self.train_folder
            elif idx < test_split:
                save_folder = self.test_folder
            else:
                save_folder = self.val_folder

            # Perform augmentation (only on image, mask remains unchanged but copied)
            self.augment_and_save(img_path, mask_path, save_folder)

        print("Data augmentation completed")


def get_folder_size(folder_path):
    """
    Calculate the total size of a folder in megabytes (MB).
    """
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            total_size += os.path.getsize(file_path)
    return round(total_size / (1024 * 1024), 2)


def count_files(folder_path):
    """
    Count the number of files in a given folder.
    """
    return sum(len(files) for _, _, files in os.walk(folder_path))


def check_image_properties(images_folder, num_samples=5):
    """
    Checks properties of random images (pixel values, shape, dtype).
    """
    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(".png") or f.endswith(".jpg")])

    if len(image_files) == 0:
        print("No images found in:", images_folder)
        return

    print(f"\nChecking {num_samples} random images in {images_folder}:")
    sampled_images = random.sample(image_files, min(num_samples, len(image_files)))

    for img_file in sampled_images:
        img_path = os.path.join(images_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        print(f"   - {img_file}: Shape={img.shape}, Type={img.dtype}, Min={img.min()}, Max={img.max()}")


def check_mask_properties(masks_folder, num_samples=5):
    """
    Checks the properties of random masks (pixel values, shape, and data type).
    """
    mask_files = sorted([f for f in os.listdir(masks_folder) if f.endswith(".png")])

    if len(mask_files) == 0:
        print("No masks found in:", masks_folder)
        return

    print(f"\nChecking {num_samples} random masks in {masks_folder}:")
    sampled_masks = random.sample(mask_files, min(num_samples, len(mask_files)))

    for mask_file in sampled_masks:
        mask_path = os.path.join(masks_folder, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        unique_values = np.unique(mask)
        print(f"   - {mask_file}: Shape={mask.shape}, Type={mask.dtype}, Unique values={unique_values}")

        if not np.array_equal(unique_values, [0, 255]) and len(unique_values) > 2:
            print(f"   Warning: Mask {mask_file} has non-binary values!")


def check_dataset(dataset_path):
    """
    Print the size, number of files, and check properties of images and masks in train, test, and val folders.
    """
    sets = ["train", "test", "val"]

    print("\nDataset Folder Size Summary:")
    for s in sets:
        images_path = os.path.join(dataset_path, s, "images")
        masks_path = os.path.join(dataset_path, s, "masks")

        if not os.path.exists(images_path) or not os.path.exists(masks_path):
            print(f"Folder missing: {s}")
            continue

        img_size = get_folder_size(images_path)
        mask_size = get_folder_size(masks_path)
        img_count = count_files(images_path)
        mask_count = count_files(masks_path)

        print(f"\n{s.upper()} SET:")
        print(f"   Images: {img_count} files ({img_size} MB)")
        print(f"   Masks: {mask_count} files ({mask_size} MB)")

        check_image_properties(images_path)
        check_mask_properties(masks_path)


if __name__ == '__main__':
    augmentor = DataAugmentation(input_folder="../Cropped+Mask/", output_folder="../Augmented_Data/", num_augmentations=5)
    augmentor.process_dataset()
    dataset_path = "../Augmented_Data/"
    check_dataset(dataset_path)
