#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the SMART_TRASH_COMPARISON application.
This file handles the initialization of the selected backend and runs the simulation or web interface.
"""

import os
import sys
import logging
from datetime import datetime

from config import ACTIVE_BACKEND, DEBUG_MODE, LOG_LEVEL
from core.simulation import SimulationEngine
from backends import get_backend

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join('logs', f'{ACTIVE_BACKEND}_run_{timestamp}.log')

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=log_format,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Initialize and run the application with the configured backend."""
    logger.info(f"Starting SMART_TRASH_COMPARISON with {ACTIVE_BACKEND} backend")
    
    # Initialize the selected backend
    backend = get_backend(ACTIVE_BACKEND)
    if not backend:
        logger.error(f"Failed to initialize {ACTIVE_BACKEND} backend")
        return
    
    # Initialize and run the simulation engine
    simulation = SimulationEngine(backend)
    simulation.run()
    
    logger.info("Application completed successfully")

if __name__ == "__main__":
    main()
