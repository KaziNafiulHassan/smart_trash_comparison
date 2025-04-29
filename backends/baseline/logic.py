#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseline backend implementation.
This module implements a simple rule-based backend using dictionary lookups.
"""

import json
import time
import logging
import sqlite3
from pathlib import Path

from backends.base_backend import BaseBackend

logger = logging.getLogger(__name__)

class BaselineBackend(BaseBackend):
    """
    Baseline backend implementation using simple lookups.

    This backend uses a static dictionary of rules loaded from a JSON file
    to provide waste sorting suggestions. It does not adapt based on user
    interactions.
    """

    def __init__(self):
        """Initialize the baseline backend."""
        super().__init__()
        self.rules = {}
        self.feedback_templates = {}
        self.db_conn = None
        logger.info(f"Created {self.name} instance")

    def initialize(self, config: dict):
        """
        Initialize the baseline backend with the provided configuration.

        Args:
            config (dict): Configuration parameters including the path to the rules file
        """
        logger.info(f"Initializing {self.name} with config: {config}")

        # Load rules from JSON file
        rule_file_path = config.get('RULE_FILE_PATH', 'data/magdeburg_rules.json')
        self._load_rules(rule_file_path)

        # Define feedback templates
        self._initialize_feedback_templates()

        # Optionally connect to SQLite database if specified in config
        if config.get('USE_SQLITE', False):
            db_path = config.get('DB_PATH', 'data/knowledge_base.db')
            self._connect_to_db(db_path)

        logger.info(f"{self.name} initialized with {len(self.rules)} rules")

    def _load_rules(self, rule_file_path: str):
        """
        Load waste sorting rules from a JSON file.

        Args:
            rule_file_path (str): Path to the JSON file containing the rules
        """
        try:
            with open(rule_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Process the rules into a flat dictionary for easy lookup
            # The file can be in different formats:
            # 1. List of dictionaries with ItemID, CorrectBin_Magdeburg, etc.
            # 2. Dictionary with ItemID as keys and bin information as values

            if isinstance(data, list):
                # Format 1: List of dictionaries
                for item in data:
                    if 'ItemID' in item and 'CorrectBin_Magdeburg' in item:
                        item_id = item['ItemID'].lower()
                        self.rules[item_id] = {
                            'CorrectBin_Magdeburg': item['CorrectBin_Magdeburg'],
                            'SpecialInstructions': item.get('Notes_EN', '')
                        }
            elif isinstance(data, dict):
                # Format 2: Dictionary with ItemID as keys
                for item_id, item_data in data.items():
                    item_id = item_id.lower()
                    if isinstance(item_data, dict) and 'CorrectBin_Magdeburg' in item_data:
                        self.rules[item_id] = item_data
                    elif isinstance(item_data, str):
                        # Simple format where value is just the bin name
                        self.rules[item_id] = {'CorrectBin_Magdeburg': item_data}

            logger.info(f"Loaded {len(self.rules)} rules from {rule_file_path}")

        except Exception as e:
            logger.error(f"Failed to load rules from {rule_file_path}: {str(e)}")
            # Initialize with empty rules if file loading fails
            self.rules = {}

    def _initialize_feedback_templates(self):
        """Initialize static feedback templates for different bin types."""
        self.feedback_templates = {
            # Templates for different bin types
            "Blue Bin": "This item belongs in the Blue Bin for paper and cardboard.",
            "Yellow Bin": "This item belongs in the Yellow Bin for packaging materials like plastic and metal.",
            "Brown Bin": "This item belongs in the Brown Bin for organic waste.",
            "Black Bin": "This item belongs in the Black Bin for residual waste.",
            "Glass Container": "This item should be taken to a glass recycling container.",
            "Hazardous Waste": "This item should be disposed of as hazardous waste at a collection point.",
            "E-Waste": "This item is electronic waste and should be taken to an e-waste collection point.",

            # Default template for unknown items
            "Unknown": "I'm not sure how to dispose of this item. Please check local guidelines or dispose in residual waste."
        }

        # Add item-specific templates for common items
        item_templates = {
            "plastic_bottle": "Plastic bottles should be empty, clean, and placed in the Yellow Bin.",
            "newspaper": "Newspapers should be placed in the Blue Bin for paper recycling.",
            "glass_bottle": "Glass bottles should be sorted by color and placed in the appropriate glass container.",
            "food_waste": "Food waste belongs in the Brown Bin for organic waste.",
            "battery": "Batteries are hazardous waste and should be taken to a collection point."
        }

        self.feedback_templates.update(item_templates)
        logger.debug(f"Initialized {len(self.feedback_templates)} feedback templates")

    def _connect_to_db(self, db_path: str):
        """
        Connect to an SQLite database (optional).

        Args:
            db_path (str): Path to the SQLite database
        """
        try:
            # Ensure directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self.db_conn = sqlite3.connect(db_path)
            self.db_conn.row_factory = sqlite3.Row

            # Create tables if they don't exist
            cursor = self.db_conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY,
                    item_id TEXT NOT NULL,
                    true_bin TEXT NOT NULL,
                    suggested_bin TEXT NOT NULL,
                    chosen_bin TEXT NOT NULL,
                    user_knowledge_level TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.db_conn.commit()

            logger.info(f"Connected to SQLite database at {db_path}")

        except Exception as e:
            logger.error(f"Failed to connect to SQLite database at {db_path}: {str(e)}")
            self.db_conn = None

    def get_sorting_suggestion(self, item_id: str, user_profile: dict) -> tuple[str, str, float]:
        """
        Get a sorting suggestion for the given item.

        Args:
            item_id (str): The identified item ID
            user_profile (dict): Information about the user (unused in baseline)

        Returns:
            tuple[str, str, float]: A tuple containing:
                - suggested_bin (str): The suggested bin for disposal
                - feedback_text (str): Explanatory text for the user
                - processing_latency_ms (float): Processing time in milliseconds
        """
        # Start timer for latency measurement
        start_time = time.time()

        # Look up the item in our rules
        item_data = self.rules.get(item_id)

        if item_data and 'CorrectBin_Magdeburg' in item_data:
            # Get the correct bin from the rules
            correct_bin = item_data['CorrectBin_Magdeburg']

            # Get feedback text - first try item-specific template, then bin-specific template
            feedback_text = self.feedback_templates.get(item_id)
            if not feedback_text:
                feedback_text = self.feedback_templates.get(correct_bin,
                                f"This item should be disposed of in the {correct_bin}.")

            # Add any special instructions from the rules
            if 'SpecialInstructions' in item_data and item_data['SpecialInstructions']:
                feedback_text += f" {item_data['SpecialInstructions']}"

        else:
            # Handle unknown items
            correct_bin = "Black Bin"  # Default to residual waste for unknown items
            feedback_text = self.feedback_templates["Unknown"]

        # Calculate latency
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Log the suggestion
        logger.debug(f"Suggested {correct_bin} for {item_id} in {latency_ms:.2f}ms")

        return correct_bin, feedback_text, latency_ms

    def record_interaction(self, item_id: str, true_bin: str, suggested_bin: str,
                          chosen_bin: str, user_profile: dict):
        """
        Record details of an interaction.

        For the baseline backend, this simply logs the interaction but doesn't
        use it to adapt future suggestions.

        Args:
            item_id (str): The identified item ID
            true_bin (str): The correct bin for the item (ground truth)
            suggested_bin (str): The bin suggested by the backend
            chosen_bin (str): The bin actually chosen by the user
            user_profile (dict): Information about the user
        """
        # Log the interaction
        logger.info(f"Interaction: item={item_id}, true={true_bin}, suggested={suggested_bin}, chosen={chosen_bin}")

        # If we have a database connection, store the interaction
        if self.db_conn:
            try:
                knowledge_level = user_profile.get('knowledge_level', 'unknown')

                cursor = self.db_conn.cursor()
                cursor.execute('''
                    INSERT INTO interactions
                    (item_id, true_bin, suggested_bin, chosen_bin, user_knowledge_level)
                    VALUES (?, ?, ?, ?, ?)
                ''', (item_id, true_bin, suggested_bin, chosen_bin, knowledge_level))

                self.db_conn.commit()
                logger.debug(f"Recorded interaction in database for {item_id}")

            except Exception as e:
                logger.error(f"Failed to record interaction in database: {str(e)}")

    def shutdown(self):
        """
        Release resources and perform cleanup before shutdown.

        Closes the database connection if one exists.
        """
        if self.db_conn:
            try:
                self.db_conn.close()
                logger.info("Closed database connection")
            except Exception as e:
                logger.error(f"Error closing database connection: {str(e)}")

        logger.info(f"{self.name} shutdown complete")
