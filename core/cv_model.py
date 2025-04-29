#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Computer Vision model for waste classification.
This module loads a pre-trained/fine-tuned CV model and provides a simple interface for prediction.
"""

import os
import json
import logging
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

from config import CV_MODEL_PATH, CV_MODEL_TYPE, CV_DEVICE

logger = logging.getLogger(__name__)

class CVModelInterface:
    """
    Interface for loading and using a pre-trained computer vision model for waste classification.
    This class does not perform training, only inference.
    """

    def __init__(self, model_path=CV_MODEL_PATH, device=CV_DEVICE):
        """
        Initialize the CV model interface.

        Args:
            model_path (str): Path to the saved model weights file
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model = None
        self.class_mapping = {}
        self.transform = None

        # Load the model and class mapping
        self._load_model()
        self._load_class_mapping()

        logger.info(f"CV model interface initialized with model from {model_path} on {self.device}")

    def _load_model(self):
        """Load the pre-trained waste classification model."""
        try:
            # Set up the preprocessing transform
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # Load the appropriate model architecture based on the model type
            if CV_MODEL_TYPE == 'resnet':
                # Initialize ResNet model
                self.model = models.resnet50(pretrained=False)

                # Load model checkpoint to determine number of classes
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # If the checkpoint contains the full model
                if 'model' in checkpoint:
                    num_classes = checkpoint['model'].fc.out_features
                    self.model = checkpoint['model']
                # If the checkpoint is just the state dict
                else:
                    # Try to determine number of classes from state dict
                    fc_weight_key = 'fc.weight'
                    if fc_weight_key in checkpoint:
                        num_classes = checkpoint[fc_weight_key].size(0)
                    else:
                        # Default to a reasonable number if we can't determine
                        num_classes = 9  # Assuming 9 waste categories
                        logger.warning(f"Could not determine number of classes from checkpoint, using default: {num_classes}")

                    # Modify the final layer for our classification task
                    num_ftrs = self.model.fc.in_features
                    self.model.fc = torch.nn.Linear(num_ftrs, num_classes)

                    # Load the state dict
                    self.model.load_state_dict(checkpoint)

                self.model.to(self.device)
                self.model.eval()
                logger.info("ResNet model loaded successfully")

            elif CV_MODEL_TYPE == 'efficientnet':
                # Initialize EfficientNet model
                self.model = models.efficientnet_b0(pretrained=False)

                # Load model checkpoint to determine number of classes
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # If the checkpoint contains the full model
                if 'model' in checkpoint:
                    num_classes = checkpoint['model'].classifier[1].out_features
                    self.model = checkpoint['model']
                # If the checkpoint is just the state dict
                else:
                    # Try to determine number of classes from state dict
                    classifier_weight_key = 'classifier.1.weight'
                    if classifier_weight_key in checkpoint:
                        num_classes = checkpoint[classifier_weight_key].size(0)
                    else:
                        # Default to a reasonable number if we can't determine
                        num_classes = 9  # Assuming 9 waste categories
                        logger.warning(f"Could not determine number of classes from checkpoint, using default: {num_classes}")

                    # Modify the final layer for our classification task
                    num_ftrs = self.model.classifier[1].in_features
                    self.model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)

                    # Load the state dict
                    self.model.load_state_dict(checkpoint)

                self.model.to(self.device)
                self.model.eval()
                logger.info("EfficientNet model loaded successfully")

            elif CV_MODEL_TYPE == 'yolo':
                try:
                    # Try to import YOLO from ultralytics
                    from ultralytics import YOLO
                    self.model = YOLO(self.model_path)
                    logger.info("YOLO model loaded successfully")
                except ImportError:
                    logger.error("Failed to import YOLO from ultralytics. Please install with: pip install ultralytics")
                    raise

            else:
                logger.error(f"Unsupported model type: {CV_MODEL_TYPE}")
                raise ValueError(f"Unsupported model type: {CV_MODEL_TYPE}")

        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {str(e)}")
            raise

    def _load_class_mapping(self):
        """
        Load the mapping from model output indices to ItemID strings.
        This could be loaded from a JSON file saved during training or defined here.
        """
        try:
            # Try to load class mapping from a JSON file next to the model
            mapping_path = os.path.join(os.path.dirname(self.model_path), 'class_mapping.json')
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    self.class_mapping = json.load(f)
                logger.info(f"Loaded class mapping from {mapping_path}")
            else:
                # Define a default mapping if no file is found
                self.class_mapping = {
                    0: "paper",
                    1: "cardboard",
                    2: "plastic",
                    3: "metal",
                    4: "glass",
                    5: "organic",
                    6: "e-waste",
                    7: "hazardous",
                    8: "mixed"
                }
                logger.warning(f"No class mapping file found at {mapping_path}, using default mapping")

                # Save the default mapping for future use
                os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
                with open(mapping_path, 'w') as f:
                    json.dump(self.class_mapping, f, indent=2)
                logger.info(f"Saved default class mapping to {mapping_path}")

        except Exception as e:
            logger.error(f"Failed to load class mapping: {str(e)}")
            # Fall back to a minimal default mapping
            self.class_mapping = {i: f"class_{i}" for i in range(9)}

    def predict(self, image_path: str) -> tuple[str, float]:
        """
        Predict the waste class for an image.

        Args:
            image_path (str): Path to the image file

        Returns:
            tuple[str, float]: A tuple containing the predicted ItemID (string) and the confidence score (float)
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Handle different model types
            if CV_MODEL_TYPE in ['resnet', 'efficientnet']:
                # Load and preprocess the image
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)

                # Perform inference
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    probabilities = F.softmax(outputs, dim=1)[0]

                # Get the top prediction
                confidence, predicted_idx = torch.max(probabilities, 0)
                predicted_idx = predicted_idx.item()
                confidence = confidence.item()

                # Map the predicted index to the corresponding ItemID
                predicted_class = self.class_mapping.get(str(predicted_idx), self.class_mapping.get(predicted_idx, f"unknown_{predicted_idx}"))

                logger.debug(f"Prediction for {image_path}: class={predicted_class}, confidence={confidence:.4f}")
                return predicted_class, confidence

            elif CV_MODEL_TYPE == 'yolo':
                # YOLO models use a different prediction approach
                results = self.model(image_path)

                # Process YOLO results
                if results and len(results) > 0:
                    # Get the highest confidence detection
                    result = results[0]
                    if len(result.boxes) > 0:
                        # Get the box with highest confidence
                        box = result.boxes[0]
                        predicted_idx = int(box.cls.item())
                        confidence = box.conf.item()

                        # Map the predicted index to the corresponding ItemID
                        predicted_class = self.class_mapping.get(str(predicted_idx), self.class_mapping.get(predicted_idx, f"unknown_{predicted_idx}"))

                        logger.debug(f"YOLO prediction for {image_path}: class={predicted_class}, confidence={confidence:.4f}")
                        return predicted_class, confidence

                # If no detection or empty results
                logger.warning(f"No detection for {image_path} using YOLO model")
                return "unknown", 0.0

            else:
                raise ValueError(f"Unsupported model type for prediction: {CV_MODEL_TYPE}")

        except Exception as e:
            logger.error(f"Error predicting image {image_path}: {str(e)}")
            raise
