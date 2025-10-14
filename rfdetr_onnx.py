# ------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 PierreMarieCurie
#
# Portions of this file are adapted from RF-DETR
# Copyright (c) Roboflow
# Licensed under the Apache License, Version 2.0
# See https://www.apache.org/licenses/LICENSE-2.0 for details.
# ------------------------------------------------------------------------

import io
import requests
import onnxruntime as ort
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random

DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MAX_NUMBER_BOXES = 300

def open_image(path):
    # Check if the path is a URL (starts with 'http://' or 'https://')
    if path.startswith('http://') or path.startswith('https://'):
        img = Image.open(io.BytesIO(requests.get(path).content))
    # If it's a local file path, open the image directly
    else:
        if os.path.exists(path):
            img = Image.open(path)
        else:
            raise FileNotFoundError(f"The file {path} does not exist.")
    return img

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def box_cxcywh_to_xyxyn(x):
    cx, cy, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    return np.stack([xmin, ymin, xmax, ymax], axis=-1)

class RFDETR_ONNX:
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    def __init__(self, onnx_model_path):
        try:
            # Load the ONNX model and initialize the ONNX Runtime session
            self.ort_session = ort.InferenceSession(onnx_model_path)

            # Get input shape
            input_info = self.ort_session.get_inputs()[0]
            self.input_height, self.input_width = input_info.shape[2:]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ONNX model from '{onnx_model_path}'. "
                f"Ensure the path is correct and the model is a valid ONNX file."
            ) from e

    def _preprocess(self, image):
        """Preprocess the input image for inference."""
        
        # Resize the image to the model's input size
        image = image.resize((self.input_width, self.input_height))

        # Convert image to numpy array and normalize pixel values
        image = np.array(image).astype(np.float32) / 255.0

        # Normalize
        image = ((image - self.MEANS) / self.STDS).astype(np.float32)

        # Change dimensions from HWC to CHW
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def _post_process(self, outputs, origin_height, origin_width, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, max_number_boxes=DEFAULT_MAX_NUMBER_BOXES):
        """
        Post-process the model's output to extract bounding boxes and class information.
        Inspired by the PostProcess class in rfdetr/lwdetr.py: https://github.com/roboflow/rf-detr/blob/1.3.0/rfdetr/models/lwdetr.py#L701
        """
        # Get masks if instance segmentation
        if len(outputs) == 3:  
            masks = outputs[2]
        else:
            masks = None
        
        # Apply sigmoid activation
        prob = sigmoid(outputs[1]) 
        
        # Get detections with highest confidence and limit to max_number_boxes
        scores = np.max(prob, axis=2).squeeze()
        labels = np.argmax(prob, axis=2).squeeze()
        sorted_idx = np.argsort(scores)[::-1]
        scores = scores[sorted_idx][:max_number_boxes]
        labels = labels[sorted_idx][:max_number_boxes]
        boxes = outputs[0].squeeze()[sorted_idx][:max_number_boxes]
        if masks is not None:
            masks = masks.squeeze()[sorted_idx][:max_number_boxes]
        
        # Convert boxes from cxcywh to xyxyn format and scale to image size (i.e xyxyn -> xyxy)
        boxes = box_cxcywh_to_xyxyn(boxes)
        boxes[..., [0, 2]] *= origin_width
        boxes[..., [1, 3]] *= origin_height
        
        # Resize the masks to the original image size if available
        if masks is not None:
            new_w, new_h = origin_width, origin_height
            masks = np.stack([
                np.array(Image.fromarray(img).resize((new_w, new_h)))
                for img in masks
            ], axis=0)
            masks = (masks > 0).astype(np.uint8) * 255 
        
        # Filter detections based on the confidence threshold
        confidence_mask = scores > confidence_threshold
        scores = scores[confidence_mask]
        labels = labels[confidence_mask]
        boxes = boxes[confidence_mask]
        if masks is not None:
            masks = masks[confidence_mask]
        
        return scores, labels, boxes, masks

    def predict(self, image_path, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, max_number_boxes=DEFAULT_MAX_NUMBER_BOXES):
        """Run the model inference and return the raw outputs."""
        
        # Load the image
        image = open_image(image_path).convert('RGB')
        origin_width, origin_height = image.size
        
        # Preprocess the image
        input_image = self._preprocess(image)

        # Get input name from the model
        input_name = self.ort_session.get_inputs()[0].name

        # Run the model
        outputs = self.ort_session.run(None, {input_name: input_image})
        
        # Post-process
        return self._post_process(outputs, origin_height, origin_width, confidence_threshold, max_number_boxes)

    def save_detections(self, image_path, boxes, labels, masks, save_image_path):
        """Draw bounding boxes, masks and class labels on the original image."""
        
        # Load base image
        base = open_image(image_path).convert("RGBA")
        result = base.copy()  # start with the base image

        # Generate a color for each unique label (RGBA)
        label_colors = {
            label: (random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                    100)  # alpha for mask
            for label in np.unique(labels)
        }

        # Loop over all masks
        if masks is not None:
            for i in range(masks.shape[0]):
                label = labels[i]
                color = label_colors[label]

                # --- Draw mask ---
                mask_overlay = Image.fromarray(masks[i]).convert("L")
                mask_overlay = ImageOps.autocontrast(mask_overlay)
                overlay_color = Image.new("RGBA", base.size, color)
                overlay_masked = Image.new("RGBA", base.size)
                overlay_masked.paste(overlay_color, (0, 0), mask_overlay)
                result = Image.alpha_composite(result, overlay_masked)

        # Convert to RGB for drawing boxes and text
        result_rgb = result.convert("RGB")
        draw = ImageDraw.Draw(result_rgb)
        font = ImageFont.load_default()

        # Loop over boxes and draw
        for i, box in enumerate(boxes.astype(int)):
            label = labels[i]
            # Use same color as mask but fully opaque for the outline
            box_color = tuple(label_colors[label][:3])  # ignore alpha
            draw.rectangle(box.tolist(), outline=box_color, width=4)

            # Draw label text
            text_x = box[0] + 5
            text_y = box[1] + 5
            draw.text((text_x, text_y), str(label), fill=box_color, font=font)

        # Show or save
        result_rgb.save(save_image_path)