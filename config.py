#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration settings for the SMART_TRASH_COMPARISON application.
This file contains all configurable parameters for the application.
"""

import os

# File paths
RULE_FILE_PATH = 'data/magdeburg_rules.json'
SCENARIO_FILE_PATH = 'data/simulation_scenarios.json'
CV_MODEL_PATH = 'models/fine_tuned_resnet_magdeburg.pth'
LOG_DIR = 'logs/'

# Simulation parameters
ACTIVE_BACKEND = 'graph'  # Options: 'baseline', 'foundation', 'graph'
CV_DEVICE = 'cuda'  # Options: 'cuda', 'cpu'
CV_MODEL_TYPE = 'resnet'  # Options: 'resnet', 'efficientnet', 'yolo'
CV_CONFIDENCE_THRESHOLD = 0.7
LOG_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
NUM_SIMULATION_RUNS = 100
DEFAULT_LANGUAGE = 'en'  # Options: 'en', 'de'

# Backend-specific configurations

# Baseline backend configuration
BASELINE_CONFIG = {
    'RULE_FILE_PATH': RULE_FILE_PATH,
    'USE_SQLITE': True,
    'DB_PATH': 'data/knowledge_base.db'
}

# Neo4j configuration for Graph backend
NEO4J_CONFIG = {
    'uri': 'neo4j://localhost:7687',
    'user': 'neo4j',
    'password': os.environ.get('NEO4J_PASSWORD', 'password')  # Use environment variable or default
}

# Foundation Model configuration
FOUNDATION_MODEL_CONFIG = {
    'model_type': 'openai_gpt4',  # Options: 'openai_gpt4', 'google_gemini', 'anthropic_claude', 'local_llm'
    'api_key_env_var': 'OPENAI_API_KEY',
    'endpoint': 'https://api.openai.com/v1/chat/completions',
    'model_name': 'gpt-4',
    'temperature': 0.3,
    'max_tokens': 500,
    'RULE_FILE_PATH': RULE_FILE_PATH
}

# Google Gemini configuration (alternative foundation model)
GEMINI_CONFIG = {
    'model_type': 'google_gemini',
    'api_key_env_var': 'GOOGLE_API_KEY',
    'model_name': 'gemini-pro',
    'RULE_FILE_PATH': RULE_FILE_PATH
}

# Anthropic Claude configuration (alternative foundation model)
CLAUDE_CONFIG = {
    'model_type': 'anthropic_claude',
    'api_key_env_var': 'ANTHROPIC_API_KEY',
    'model_name': 'claude-3-opus-20240229',
    'RULE_FILE_PATH': RULE_FILE_PATH
}

# Local LLM configuration (alternative foundation model)
LOCAL_LLM_CONFIG = {
    'model_type': 'local_llm',
    'model_path': 'models/local_llm_model',
    'RULE_FILE_PATH': RULE_FILE_PATH
}

# Debug mode
DEBUG_MODE = True

# Additional paths (for backward compatibility)
DATA_DIR = 'data'
MODELS_DIR = 'models'
LOGS_DIR = LOG_DIR
