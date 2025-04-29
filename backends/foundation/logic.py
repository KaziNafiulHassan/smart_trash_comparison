#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Foundation backend implementation.
This module implements a backend using foundation models (LLMs) for waste classification.
"""

import json
import time
import logging
import os
from typing import Optional

# Import OpenAI for API integration
import openai

from backends.base_backend import BaseBackend

logger = logging.getLogger(__name__)

class FoundationBackend(BaseBackend):
    """
    Foundation backend implementation using LLMs.

    This backend uses foundation models (LLMs/VLMs) to generate feedback text
    for waste sorting suggestions. It looks up the correct bin from rules and
    uses the LLM to generate helpful, personalized feedback.
    """

    def __init__(self):
        """Initialize the foundation backend."""
        super().__init__()
        self.rules = []  # Will be loaded in initialize()
        self.model_type = None
        self.model_config = {}
        self.client = None  # Will be initialized if using OpenAI
        logger.info("Foundation backend instance created")

    def initialize(self, config: dict):
        """
        Initialize the foundation backend with the provided configuration.

        Args:
            config (dict): Configuration parameters including rule file path and model settings
        """
        logger.info("Initializing Foundation backend")

        # Load rules from the specified path
        rule_file_path = config.get('RULE_FILE_PATH')
        if not rule_file_path:
            logger.error("No rule file path specified in config")
            raise ValueError("RULE_FILE_PATH must be specified in config")

        self._load_rules(rule_file_path)

        # Read foundation model configuration
        foundation_model_config = config.get('FOUNDATION_MODEL_CONFIG', {})
        if not foundation_model_config:
            logger.warning("No foundation model configuration found, using defaults")
            foundation_model_config = {
                'model_type': 'openai_gpt4',
                'api_key_env_var': 'OPENAI_API_KEY',
                'endpoint': 'https://api.openai.com/v1/chat/completions',
                'model_name': 'gpt-4'
            }

        # Store relevant settings
        self.model_type = foundation_model_config.get('model_type', 'unknown')
        self.model_config = foundation_model_config

        # Initialize OpenAI client if using OpenAI
        if self.model_type.startswith('openai_'):
            try:
                # Get API key from environment variable
                api_key_env_var = foundation_model_config.get('api_key_env_var', 'OPENAI_API_KEY')
                api_key = os.environ.get(api_key_env_var)

                if not api_key:
                    logger.warning(f"OpenAI API key not found in environment variable {api_key_env_var}")
                    logger.warning("Using empty API key, which will likely cause API calls to fail")

                # Initialize OpenAI client
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                self.client = None

        # Print message indicating which model type is configured
        logger.info(f"Foundation Backend configured for {self.model_type}")
        print(f"Foundation Backend configured for {self.model_type}")

    def _load_rules(self, rule_file_path: str):
        """
        Load waste sorting rules from a JSON file.

        Args:
            rule_file_path (str): Path to the JSON file containing the rules
        """
        try:
            if not os.path.exists(rule_file_path):
                logger.error(f"Rule file not found: {rule_file_path}")
                raise FileNotFoundError(f"Rule file not found: {rule_file_path}")

            with open(rule_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Store the rules as a list or convert to a dictionary keyed by ItemID
            if isinstance(data, list):
                self.rules = data
                logger.info(f"Loaded {len(self.rules)} rules from {rule_file_path}")
            else:
                logger.error(f"Expected a JSON array in {rule_file_path}, but got {type(data)}")
                raise ValueError(f"Expected a JSON array in {rule_file_path}")

        except Exception as e:
            logger.error(f"Failed to load rules from {rule_file_path}: {str(e)}")
            raise

    def _find_rule_by_item_id(self, item_id: str):
        """
        Find a rule by item ID.

        Args:
            item_id (str): The item ID to look up

        Returns:
            dict: The rule for the item, or None if not found
        """
        for rule in self.rules:
            if rule.get('ItemID', '').lower() == item_id.lower():
                return rule
        return None

    def _generate_openai_feedback(self, item_name: str, correct_bin: str, user_profile: Optional[dict] = None) -> str:
        """
        Generate feedback text using OpenAI API.

        Args:
            item_name (str): The name of the item
            correct_bin (str): The correct bin for the item
            user_profile (dict, optional): Information about the user

        Returns:
            str: Generated feedback text
        """
        if not self.client:
            logger.warning("OpenAI client not initialized, returning generic feedback")
            return f"{item_name} belongs in the {correct_bin}."

        try:
            # Extract relevant information from user profile
            user_context = ""
            if user_profile:
                # Check if user has made mistakes with this item before
                if 'previous_errors' in user_profile and item_name in user_profile['previous_errors']:
                    previous_wrong_bin = user_profile['previous_errors'][item_name]
                    user_context = f"The user has previously mistakenly put this item in the {previous_wrong_bin}. "

                # Check user's knowledge level
                if 'knowledge_level' in user_profile:
                    knowledge_level = user_profile['knowledge_level']
                    if knowledge_level == 'beginner':
                        user_context += "The user is a beginner in waste sorting. "
                    elif knowledge_level == 'expert':
                        user_context += "The user is experienced in waste sorting. "

            # Construct the prompt
            prompt = f"""You are a helpful waste sorting assistant for Magdeburg, Germany.

Item: {item_name}
Correct Bin: {correct_bin}
{user_context}

Generate concise, helpful, and encouraging feedback explaining *why* this item belongs in the '{correct_bin}' or offering a relevant sorting tip. Tailor the feedback slightly if previous mistakes are noted. Respond only with the feedback text.
"""

            # Get model name from config
            model_name = self.model_config.get('model_name', 'gpt-4')

            # Make API call
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a waste management expert helping users properly dispose of items in Magdeburg, Germany."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            # Extract feedback text from response
            feedback = response.choices[0].message.content.strip()

            # Log success
            logger.debug(f"Generated feedback for {item_name} using OpenAI API")

            return feedback

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return f"{item_name} belongs in the {correct_bin}."
        except openai.APITimeoutError as e:
            logger.error(f"OpenAI API timeout: {str(e)}")
            return f"{item_name} belongs in the {correct_bin}."
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API connection error: {str(e)}")
            return f"{item_name} belongs in the {correct_bin}."
        except Exception as e:
            logger.error(f"Error generating feedback with OpenAI: {str(e)}")
            return f"{item_name} belongs in the {correct_bin}."

    def get_sorting_suggestion(self, item_id: str, user_profile: dict) -> tuple[str, str, float]:
        """
        Get a sorting suggestion for the given item.

        Args:
            item_id (str): The identified item ID
            user_profile (dict): Information about the user

        Returns:
            tuple[str, str, float]: A tuple containing:
                - suggested_bin (str): The suggested bin for disposal
                - feedback_text (str): Explanatory text for the user
                - processing_latency_ms (float): Processing time in milliseconds
        """
        # Start timer for latency measurement
        start_time = time.time()

        try:
            # Look up the item in the rules
            rule = self._find_rule_by_item_id(item_id)

            if rule:
                # Get the correct bin and item name from the rule
                correct_bin = rule.get('CorrectBin_Magdeburg', 'Unknown')
                item_name = rule.get('ItemName_DE', item_id)

                # Generate feedback based on the configured model type
                if self.model_type.startswith('openai_'):
                    feedback = self._generate_openai_feedback(item_name, correct_bin, user_profile)
                else:
                    # Use placeholder feedback for other model types
                    feedback = f"Placeholder feedback for {item_name}. Correct bin: {correct_bin}. LLM call needed."
            else:
                # Handle case where item is not found
                logger.warning(f"Item ID not found in rules: {item_id}")
                correct_bin = "Unknown"
                feedback = f"Item not found: {item_id}."

        except Exception as e:
            logger.error(f"Error getting sorting suggestion for {item_id}: {str(e)}")
            correct_bin = "Unknown"
            feedback = "An error occurred while processing your request."

        # Calculate latency
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Return the bin determined from the rules lookup, not from LLM
        return correct_bin, feedback, latency_ms

    def record_interaction(self, item_id: str, true_bin: str, suggested_bin: str,
                          chosen_bin: str, user_profile: dict):
        """
        Record details of an interaction.

        Args:
            item_id (str): The identified item ID
            true_bin (str): The correct bin for the item (ground truth)
            suggested_bin (str): The bin suggested by the backend
            chosen_bin (str): The bin actually chosen by the user
            user_profile (dict): Information about the user
        """
        # This is a placeholder implementation
        pass

    def shutdown(self):
        """
        Release resources and perform cleanup before shutdown.
        """
        # This is a placeholder implementation
        pass
