"""
Lung Cancer Detection - Data Preprocessing Module
Author: ML Project Team
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class LungCancerPreprocessor:
    """
    Comprehensive preprocessing pipeline for lung cancer CT images
    """
    
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self, data_path):
        """
        Load images and labels from directory structure:
        data_path/
            Normal/
            Benign/
            Malignant/
        """
        images = []
        labels = []
        
        categories = ['Normal', 'Benign', 'Malignant']
        
        print("Loading dataset...")
        for category in categories:
            path = os.path.join(data_path, category)
            if not os.path.exists(path):
                print(f"Warning: {path} not found!")
                continue
                
            for img_name in tqdm(os.listdir(path), desc=f"Loading {category}"):
                try:
                    img_path = os.path.join(path, img_name)
                    img = cv2.imread(img_path)
                    
                    if img is not None:
                        images.append(img)
                        labels.append(category)
                except Exception as e:
                    print(f"Error loading {img_name}: {str(e)}")
                    
        print(f"Loaded {len(images)} images")
        return images, np.array(labels)
    
    def preprocess_image(self, image):
        """
        Apply preprocessing techniques:
        1. Resize
        2. Noise reduction
        3. Contrast enhancement (CLAHE)
        4. Normalization
        """
        # Resize
        img_resized = cv2.resize(image, self.img_size)
        
        # Convert to grayscale for processing
        if len(img_resized.shape) == 3:
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_resized
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Normalize to [0, 1]
        normalized = enhanced / 255.0
        
        # Convert back to 3 channels for CNN
        img_final = np.stack([normalized] * 3, axis=-1)
        
        return img_final
    
    def augment_data(self, images, labels):
        """
        Data augmentation for imbalanced datasets
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.15,
            fill_mode='nearest'
        )
        
        return datagen
    
    def prepare_data(self, images, labels, test_size=0.2, val_size=0.1):
        """
        Preprocess all images and split into train/val/test
        """
        print("Preprocessing images...")
        processed_images = []
        
        for img in tqdm(images, desc="Preprocessing"):
            processed_img = self.preprocess_image(img)
            processed_images.append(processed_img)
        
        processed_images = np.array(processed_images)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Train-test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            processed_images, labels_encoded, 
            test_size=test_size, 
            random_state=42,
            stratify=labels_encoded
        )
        
        # Train-validation split
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=42,
            stratify=y_temp
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def visualize_samples(self, images, labels, n_samples=9):
        """
        Visualize sample images from each class
        """
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(min(n_samples, len(images))):
            axes[i].imshow(images[i])
            axes[i].set_title(f"Label: {labels[i]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_distribution(self, labels):
        """
        Plot class distribution
        """
        plt.figure(figsize=(10, 6))
        unique, counts = np.unique(labels, return_counts=True)
        
        sns.barplot(x=unique, y=counts)
        plt.title('Class Distribution', fontsize=16)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        for i, count in enumerate(counts):
            plt.text(i, count + 5, str(count), ha='center', fontsize=12)
        
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    preprocessor = LungCancerPreprocessor(img_size=(224, 224))

    #  CORRECT DATA PATH
    data_path = "../data/raw"

    images, labels = preprocessor.load_dataset(data_path)

    preprocessor.visualize_samples(images, labels)
    preprocessor.plot_class_distribution(labels)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        preprocessor.prepare_data(images, labels)

    # SAVE IN CLEAN LOCATION
    os.makedirs("../data/processed", exist_ok=True)

    np.save("../data/processed/X_train.npy", X_train)
    np.save("../data/processed/y_train.npy", y_train)
    np.save("../data/processed/X_val.npy", X_val)
    np.save("../data/processed/y_val.npy", y_val)
    np.save("../data/processed/X_test.npy", X_test)
    np.save("../data/processed/y_test.npy", y_test)

    print("Preprocessing complete! Data saved.")