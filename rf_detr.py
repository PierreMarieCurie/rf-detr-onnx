import io
import requests
import onnxruntime as ort
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

def open_image(path):
    # Check if the path is a URL (starts with 'http://' or 'https://')
    if path.startswith('http://') or path.startswith('https://'):
        # If it's a URL, use requests to fetch the image
        img = Image.open(io.BytesIO(requests.get(path).content))
    else:
        # If it's a local file path, open the image directly
        if os.path.exists(path):
            img = Image.open(path)
        else:
            raise FileNotFoundError(f"The file {path} does not exist.")
    
    return img

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def box_cxcywh_to_xyxy_numpy(x):
    x_c, y_c, w, h = np.split(x, 4, axis=-1)
    b = np.concatenate([
        x_c - 0.5 * np.clip(w, a_min=0.0, a_max=None),
        y_c - 0.5 * np.clip(h, a_min=0.0, a_max=None),
        x_c + 0.5 * np.clip(w, a_min=0.0, a_max=None),
        y_c + 0.5 * np.clip(h, a_min=0.0, a_max=None)
    ], axis=-1)
    return b

class RTDETR_ONNX:
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    def __init__(self, onnx_model_path):
        
        # Load the ONNX model and initialize the ONNX Runtime session
        self.ort_session = ort.InferenceSession(onnx_model_path)

        # Get input shape
        input_info = self.ort_session.get_inputs()[0]
        self.input_height, self.input_width = input_info.shape[2:]


    def _preprocess_image(self, image):
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
    
    def _post_process(self, outputs, origin_height, origin_width, confidence_threshold, max_number_boxes):
        """Post-process the model's output to extract bounding boxes and class information."""
        # Get the bounding box and class scores
        pred_boxes, pred_logits = outputs

        # Apply sigmoid activation
        prob = sigmoid(pred_logits) 

        # Get the top-k values and indices
        flat_prob = prob[0].flatten()
        topk_indexes = np.argsort(flat_prob)[-max_number_boxes:][::-1]
        topk_values = np.take_along_axis(flat_prob, topk_indexes, axis=0)
        scores = topk_values
        topk_boxes = topk_indexes // pred_logits.shape[2]
        labels = topk_indexes % pred_logits.shape[2]

        # Gather boxes corresponding to top-k indices
        boxes = box_cxcywh_to_xyxy_numpy(pred_boxes[0])
        boxes = np.take_along_axis(boxes, np.expand_dims(topk_boxes, axis=-1).repeat(4, axis=-1), axis=0)
        
        # Rescale box locations
        target_sizes = np.array([[origin_height, origin_width]])
        img_h, img_w = target_sizes[:, 0], target_sizes[:, 1]
        scale_fct = np.stack([img_w, img_h, img_w, img_h], axis=1)  
        boxes = boxes * scale_fct[0, :] 

        # Filter detections based on the confidence threshold
        high_confidence_indices = np.argmin(scores > confidence_threshold)
        scores = scores[:high_confidence_indices]
        labels = labels[:high_confidence_indices]
        boxes = boxes[:high_confidence_indices]

        return scores, labels, boxes

    def run_inference(self, image_path, confidence_threshold=0.5, max_number_boxes=100):
        """Run the model inference and return the raw outputs."""
        
        # Load the image
        image = open_image(image_path).convert('RGB')
        origin_width, origin_height = image.size
        
        # Preprocess the image
        input_image = self._preprocess_image(image)

        # Get input name from the model
        input_name = self.ort_session.get_inputs()[0].name

        # Run the model
        outputs = self.ort_session.run(None, {input_name: input_image})

        # Post-process
        return self._post_process(outputs, origin_height, origin_width, confidence_threshold, max_number_boxes)
    
    
    def save_detections(self, image_path, boxes, labels, save_image_path):
        """Draw bounding boxes and class labels on the original image."""
        # Load the original image
        image = open_image(image_path).convert('RGB')
        
        draw = ImageDraw.Draw(image)

        # Loop over the boxes
        for i, box in enumerate(boxes.astype(int)):
            
            # Draw the rectangle (box) on the image
            draw.rectangle(box.tolist(), outline="green", width=4)

            # Using default font
            font = ImageFont.load_default()

            # Position the text inside the rectangle
            text_x = box[0] + 10  # Left margin for text
            text_y = box[1] + 10  # Top margin for text
            draw.text((text_x, text_y), str(labels[i]), fill="red", font=font)

        # Save the image with the rectangle and text
        image.save(save_image_path)