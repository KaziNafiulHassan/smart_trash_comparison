#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backend module initialization.
This module provides factory functions for creating backend instances.
"""

import logging
from importlib import import_module

logger = logging.getLogger(__name__)

def get_backend(backend_name):
    """
    Factory function to get a backend instance by name.
    
    Args:
        backend_name (str): Name of the backend to create ('baseline', 'foundation', or 'graph')
        
    Returns:
        BaseBackend: An instance of the requested backend
    """
    try:
        # Map backend names to their module paths and class names
        backend_map = {
            'baseline': ('backends.baseline.logic', 'BaselineBackend'),
            'foundation': ('backends.foundation.logic', 'FoundationBackend'),
            'graph': ('backends.graph.logic', 'GraphBackend')
        }
        
        if backend_name not in backend_map:
            logger.error(f"Unknown backend: {backend_name}")
            return None
        
        module_path, class_name = backend_map[backend_name]
        
        # Import the module and get the class
        module = import_module(module_path)
        backend_class = getattr(module, class_name)
        
        # Create and return an instance of the backend
        backend = backend_class()
        logger.info(f"Created backend: {backend_name}")
        return backend
        
    except Exception as e:
        logger.error(f"Failed to create backend {backend_name}: {str(e)}")
        return None
