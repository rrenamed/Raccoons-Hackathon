import functools
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

import torch
from lang_sam import LangSAM

from torchvision.utils import draw_segmentation_masks

class ForestMask:
    def download_image(url):
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    def save_mask(mask_np, filename):
        mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_image.save(filename)

    def calculate_area():
        img = cv2.imread('forest_mask.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_area = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 0:
                total_area += cv2.contourArea(cnt)

        image_size = 50
        white_area = total_area * (1 / (thresh.shape[0] * thresh.shape[1])) * (image_size ** 2)
        
        return white_area

    def display_image_with_masks(image, mask, masks):
        
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # axes[0].imshow(image)
        # axes[0].set_title("Original Image", color='white')
        # axes[0].axis('off')
        
        image_array = np.asarray(image)
        tensor_image = torch.from_numpy(image_array).permute(2, 0, 1)
        image_with_mask = draw_segmentation_masks(tensor_image, masks=masks, colors=['cyan'] * len(masks), alpha=0.4)
        result_image = image_with_mask.numpy().transpose(1, 2, 0)
        # axes[1].imshow(result_image)
        # axes[1].set_title(f"Forest mask", color='white')
        # axes[1].axis('off')
        
        area = ForestMask.calculate_area()
        # axes[2].imshow(mask, cmap='gray')
        # axes[2].set_title(f"Area: {area:.2f} m²", color='white')
        # axes[2].axis('off')

        # plt.tight_layout()
        # plt.show()
        
        return result_image, area

    def display_image_with_boxes(image, boxes, logits):
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title("Image with Bounding Boxes")
        ax.axis('off')

        for box, logit in zip(boxes, logits):
            x_min, y_min, x_max, y_max = box
            confidence_score = round(logit.item(), 2)  # Convert logit to a scalar before rounding
            box_width = x_max - x_min
            box_height = y_max - y_min

            # Draw bounding box
            rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

            # Add confidence score as text
            ax.text(x_min, y_min, f"Confidence: {confidence_score}", fontsize=8, color='red', verticalalignment='top')

        plt.show()

    def print_bounding_boxes(boxes):
        print("Bounding Boxes:")
        for i, box in enumerate(boxes):
            print(f"Box {i+1}: {box}")

    def print_detected_phrases(phrases):
        print("\nDetected Phrases:")
        for i, phrase in enumerate(phrases):
            print(f"Phrase {i+1}: {phrase}")

    def print_logits(logits):
        print("\nConfidence:")
        for i, logit in enumerate(logits):
            print(f"Logit {i+1}: {logit}")

    def get_all(image):
        warnings.filterwarnings("ignore")

        try:
            image_pil = Image.open(image).convert("RGB")

            model = LangSAM()
            masks, boxes, phrases, logits = model.predict(image_pil, 'tree', box_threshold=0.25)

            masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
            mask = functools.reduce(lambda m0, m1: np.where(m1 == 0, m0, m1), masks_np)
            # ForestMask.save_mask(mask, "forest_mask.png")

            return_image, return_area = ForestMask.display_image_with_masks(image_pil, mask, masks)

        except (requests.exceptions.RequestException, IOError) as e:
            print(f"Error: {e}")
            
        return return_image, return_area