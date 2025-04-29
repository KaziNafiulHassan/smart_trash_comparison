#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the SMART_TRASH_COMPARISON application.
"""

import os
import json
import logging
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)

def ensure_dir_exists(directory):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory (str): Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.debug(f"Created directory: {directory}")

def load_json_file(file_path):
    """
    Load and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Parsed JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON file: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON file {file_path}: {str(e)}")
        return {}

def save_json_file(file_path, data):
    """
    Save data to a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        data: Data to save
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved JSON file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON file {file_path}: {str(e)}")

def connect_to_sqlite_db(db_path):
    """
    Connect to an SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database
        
    Returns:
        sqlite3.Connection: Database connection
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        logger.debug(f"Connected to SQLite database: {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to SQLite database {db_path}: {str(e)}")
        return None

def format_timestamp(timestamp=None):
    """
    Format a timestamp for logging or display.
    
    Args:
        timestamp (datetime, optional): Timestamp to format. Defaults to current time.
        
    Returns:
        str: Formatted timestamp
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')

def calculate_accuracy(results):
    """
    Calculate accuracy from a list of results.
    
    Args:
        results (list): List of result dictionaries with 'is_correct' field
        
    Returns:
        float: Accuracy as a percentage
    """
    if not results:
        return 0.0
    
    correct = sum(1 for r in results if r.get('is_correct', False))
    return (correct / len(results)) * 100.0

def normalize_waste_type(waste_type):
    """
    Normalize waste type strings for consistent comparison.
    
    Args:
        waste_type (str): Waste type string
        
    Returns:
        str: Normalized waste type
    """
    if not waste_type:
        return ''
    
    # Convert to lowercase and remove extra whitespace
    normalized = waste_type.lower().strip()
    
    # Map common variations to standard types
    mapping = {
        'paper': ['paper', 'newspaper', 'cardboard', 'carton'],
        'plastic': ['plastic', 'pet', 'hdpe', 'ldpe', 'pp', 'ps'],
        'glass': ['glass', 'bottle'],
        'metal': ['metal', 'aluminum', 'tin', 'steel', 'can'],
        'organic': ['organic', 'food', 'compost', 'bio', 'biodegradable'],
        'electronic': ['electronic', 'e-waste', 'battery', 'computer', 'phone'],
        'hazardous': ['hazardous', 'chemical', 'paint', 'oil', 'toxic'],
        'residual': ['residual', 'other', 'rest', 'mixed', 'general']
    }
    
    for standard, variations in mapping.items():
        if any(var in normalized for var in variations):
            return standard
    
    return normalized
