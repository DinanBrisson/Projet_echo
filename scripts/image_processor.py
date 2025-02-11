import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import os
import json
import re


class ImageProcessor:
    def __init__(self,
                 data_folder="../Data/",
                 annotated_folder="../Data_Annotated/",
                 cropped_folder_gray="../Cropped+Mask/",
                 black_ratio_threshold=0.5,
                 intensity_threshold=10,
                 delete_black_images=False):
        """
        :param data_folder: Folder containing the images and XML files.
        :param annotated_folder: Folder where the annotated images will be saved.
        :param cropped_folder_gray: Folder where the cropped grayscale images will be saved.
        :param black_ratio_threshold: The ratio of dark pixels (intensity < intensity_threshold)
                                      above which the crop is considered too dark.
        :param intensity_threshold: The intensity threshold (0-255) to consider a pixel as dark.
        :param delete_black_images: If True, deletes the original image if the crop is too dark.
        """
        self.data_folder = data_folder
        self.annotated_folder = annotated_folder
        self.cropped_folder_gray = cropped_folder_gray
        self.black_ratio_threshold = black_ratio_threshold
        self.intensity_threshold = intensity_threshold
        self.delete_black_images = delete_black_images

        os.makedirs(self.annotated_folder, exist_ok=True)
        os.makedirs(self.cropped_folder_gray, exist_ok=True)

    @staticmethod
    def natural_sort_key(s):
        """
        Returns a key for natural sorting of strings (e.g., "1_1", "1_2", "2_1" ...)
        """
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

    def is_too_black(self, img):
        """
        Checks if a crop contains too many dark pixels.
        The image is converted to grayscale and the ratio of pixels with an intensity
        lower than intensity_threshold is compared to black_ratio_threshold.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        black_pixels = np.sum(gray < self.intensity_threshold)
        ratio = black_pixels / gray.size
        return ratio > self.black_ratio_threshold

    @staticmethod
    def extract_annotations(xml_path):
        """
        Extracts annotations from an XML file.

        :param xml_path: Path to the XML file.
        :return: A dictionary of annotations.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        annotations = {}
        for mark in root.findall(".//mark"):
            image_id = mark.find("image").text.strip()
            svg_data = mark.find("svg").text
            if svg_data:
                try:
                    svg_data = json.loads(svg_data)
                    points = [(point["x"], point["y"]) for point in svg_data[0]["points"]]
                    if image_id in annotations:
                        annotations[image_id].append(points)
                    else:
                        annotations[image_id] = [points]
                except json.JSONDecodeError:
                    print(f"JSON parsing error in {xml_path}")
        return annotations

    @staticmethod
    def is_double_image_by_annotations(annotations):
        """
        Considers the image as "double" if it contains 2 annotations.

        :param annotations: Dictionary of annotations.
        :return: True if there are exactly 2 annotations, False otherwise.
        """
        if not annotations:
            return False
        key = list(annotations.keys())[0]
        return len(annotations[key]) == 2

    @staticmethod
    def get_new_base_names(base_filename):
        """
        For a file whose name follows the pattern 'X_Y' (e.g., "1_1"),
        returns:
          - left_base: unchanged (e.g., "1_1")
          - right_base: incremented (e.g., "1_2")
        If the pattern is not followed, _1 and _2 are appended.

        :param base_filename: The base filename (without extension).
        :return: A tuple (left_base, right_base).
        """
        parts = base_filename.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            left_base = base_filename
            right_base = "_".join(parts[:-1] + [str(int(parts[-1]) + 1)])
        else:
            left_base = base_filename + "_1"
            right_base = base_filename + "_2"
        return left_base, right_base

    def process_files(self):
        """
        Iterates through XML and image files in the data folder, and processes each image.
        """
        xml_files = sorted([f for f in os.listdir(self.data_folder) if f.endswith(".xml")],
                           key=self.natural_sort_key)
        image_files = sorted([f for f in os.listdir(self.data_folder)
                              if f.endswith(".jpg") or f.endswith(".png")],
                             key=self.natural_sort_key)

        # Map XML files to images using the filename prefix
        xml_to_images = {}
        for image_file in image_files:
            base_name = image_file.split("_")[0]
            xml_to_images.setdefault(base_name, []).append(image_file)

        for xml_file in xml_files:
            xml_path = os.path.join(self.data_folder, xml_file)
            base_name = xml_file.replace(".xml", "")
            annotations = self.extract_annotations(xml_path)
            if not annotations:
                print(f"No annotations found in {xml_file}.")
                continue
            if base_name not in xml_to_images:
                print(f"No image found for {xml_file}.")
                continue
            for image_file in xml_to_images[base_name]:
                self.process_image(image_file, annotations)
        print("Processing completed.")
        self.verify_image_mask_pairs() # Check if each cropped image has a mask

    def process_image(self, image_file, annotations):
        """
        Processes an image based on the number of annotations.
        The check for dark content is done only on the crop.

        :param image_file: The image filename.
        :param annotations: The annotations dictionary.
        """
        image_path = os.path.join(self.data_folder, image_file)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Unable to load {image_file}.")
            return

        base_filename = os.path.splitext(image_file)[0]
        # If the image has exactly 2 annotations, consider it as a double image.
        if self.is_double_image_by_annotations(annotations):
            left_base, right_base = self.get_new_base_names(base_filename)
            key = list(annotations.keys())[0]
            ann_left, ann_right = annotations[key]  # Retrieve the 2 annotations.
            self.crop_and_save(img, ann_left, left_base)
            self.crop_and_save(img, ann_right, right_base)
        else:
            # For a non-double image, use only the first annotation.
            self.process_image_part(img, annotations, base_filename)

    def crop_and_save(self, img, annotation, base_name):
        """
        Crops the image based on the annotation, resizes it, applies processing, and creates a corresponding mask.

        :param img: Original image.
        :param annotation: Annotation used for cropping.
        :param base_name: Base name for output files.
        """
        pts = np.array(annotation, np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        if w > 0 and h > 0:
            # Crop the image to the bounding box of the annotation
            cropped = img[y:y + h, x:x + w].copy()
            resized = cv2.resize(cropped, (256, 256))

            if self.is_too_black(resized):
                print(f"The cropped region for {base_name} is too dark, ignored.")
                return

            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            median_filtered = cv2.medianBlur(gray, 3)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            final_img = clahe.apply(median_filtered)

            filename = f"{base_name}_cropped_gray.jpg"
            path = os.path.join(self.cropped_folder_gray, filename)
            cv2.imwrite(path, final_img)
            print(f"Cropped image saved: {filename}")

            # Pass the **original image shape** to the mask function
            self.create_and_save_mask(img.shape[:2], (256, 256), annotation, base_name)

            # Display the processed image
            plt.figure(figsize=(4, 4))
            plt.imshow(final_img, cmap="gray")
            plt.axis("off")
            plt.title(base_name)
            plt.show()
        else:
            print(f"Invalid annotation for {base_name}, no cropping performed.")

    def process_image_part(self, img, annotations, base_name):
        """
        Processes a non-double image:
          1. Saves an annotated image as "{base_name}_annotated.jpg".
          2. Crops the image using the first annotation and saves it as "{base_name}_cropped_gray.jpg".

        :param img: The original image.
        :param annotations: The annotations dictionary.
        :param base_name: The base name for the saved files.
        """
        # Save the annotated image.
        img_annotated = img.copy()
        key = list(annotations.keys())[0] if annotations else None
        if key:
            # Draw all annotations on the image.
            for points in annotations[key]:
                pts = np.array(points, np.int32)
                cv2.polylines(img_annotated, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        annotated_filename = f"{base_name}_annotated.jpg"
        annotated_path = os.path.join(self.annotated_folder, annotated_filename)
        cv2.imwrite(annotated_path, img_annotated)
        print(f"Annotated image saved: {annotated_filename}")

        # Use only the first annotation for cropping.
        if key and len(annotations[key]) > 0:
            annotation = annotations[key][0]
            self.crop_and_save(img, annotation, base_name)

    def create_and_save_mask(self, original_shape, cropped_shape, annotation, base_name):
        """
        Creates a binary mask based on the annotation, resizes it, and saves it.

        :param original_shape: Tuple (height, width) of the original image.
        :param cropped_shape: Tuple (height, width) of the cropped image (e.g., (256, 256)).
        :param annotation: List of points defining the annotation.
        :param base_name: Base name for the output file.
        """
        # Create a black mask of the same size as the original image
        full_mask = np.zeros(original_shape, dtype=np.uint8)

        # Convert the annotation points into a NumPy array and draw the polygon on the mask
        pts = np.array(annotation, np.int32)
        cv2.fillPoly(full_mask, [pts], 255)  # Fill the annotated region in white

        # Extract the same region as the cropped image
        x, y, w, h = cv2.boundingRect(pts)
        if w > 0 and h > 0:
            # Crop the mask to match the selected region
            cropped_mask = full_mask[y:y + h, x:x + w].copy()

            # Resize the mask to match the cropped image dimensions (256x256)
            resized_mask = cv2.resize(cropped_mask, cropped_shape)

            # Save the mask as a PNG file
            mask_filename = f"{base_name}_mask.png"
            mask_path = os.path.join(self.cropped_folder_gray, mask_filename)
            cv2.imwrite(mask_path, resized_mask)
            print(f"Mask saved: {mask_filename}")

            # Display the mask
            plt.figure(figsize=(4, 4))
            plt.imshow(resized_mask, cmap="gray")
            plt.axis("off")
            plt.title(f"Mask for {base_name}")
            plt.show()
        else:
            print(f"Invalid annotation for {base_name}, mask not created.")

    def verify_image_mask_pairs(self):
        """
        Verifies if each cropped image has a corresponding mask in the same directory.

        - Looks for files in `self.cropped_folder_gray`
        - Checks if each `*_cropped_gray.jpg` has a corresponding `*_mask.png`
        - Prints missing masks if any
        """
        if not os.path.exists(self.cropped_folder_gray):
            print(f"The folder '{self.cropped_folder_gray}' does not exist.")
            return

        cropped_images = set()
        masks = set()

        # List all files in the cropped folder
        for file in os.listdir(self.cropped_folder_gray):
            if file.endswith("_cropped_gray.jpg"):
                cropped_images.add(file.replace("_cropped_gray.jpg", ""))
            elif file.endswith("_mask.png"):
                masks.add(file.replace("_mask.png", ""))

        # Find images without corresponding masks
        missing_masks = cropped_images - masks

        if missing_masks:
            print("\nMissing Masks for the following images:")
            for img in sorted(missing_masks):
                print(f"- {img}_cropped_gray.jpg (No {img}_mask.png found)")
        else:
            print("\nAll images have corresponding masks!")

