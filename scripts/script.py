from scripts.image_processor import ImageProcessor

if __name__ == "__main__":
    processor = ImageProcessor(delete_black_images=True)
    processor.process_files()