#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to configure and run simulations for the SMART_TRASH_COMPARISON project.
This script handles command-line arguments, configuration loading, and running the simulation.
"""

import argparse
import importlib
import logging
import os
import sys
import time
from datetime import datetime

# Import configuration
import config
from core.cv_model import CVModelInterface
from core.simulation import SimulationEngine

def setup_logging(log_file=None, backend=None, scenario_id=None):
    """
    Set up logging configuration.
    
    Args:
        log_file (str, optional): Custom log file path
        backend (str, optional): Backend name for default log file
        scenario_id (str, optional): Scenario ID for default log file
    
    Returns:
        str: Path to the log file
    """
    # Create logs directory if it doesn't exist
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Generate default log filename if not provided
    if not log_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        scenario_str = f"_{scenario_id}" if scenario_id else ""
        log_file = os.path.join(config.LOG_DIR, f"{backend}{scenario_str}_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run waste sorting simulation with different backends')
    
    parser.add_argument(
        '--backend',
        choices=['baseline', 'foundation', 'graph'],
        default=config.ACTIVE_BACKEND,
        help='Backend to use for the simulation (default: from config)'
    )
    
    parser.add_argument(
        '--scenario_id',
        type=str,
        help='ID of the scenario to run from the scenarios file'
    )
    
    parser.add_argument(
        '--log_file',
        type=str,
        help='Custom log file path'
    )
    
    return parser.parse_args()

def load_backend(backend_name):
    """
    Dynamically import and instantiate the selected backend.
    
    Args:
        backend_name (str): Name of the backend to load
    
    Returns:
        BaseBackend: Instantiated backend
    """
    try:
        # Map backend names to their module paths and class names
        backend_map = {
            'baseline': ('backends.baseline.logic', 'BaselineBackend'),
            'foundation': ('backends.foundation.logic', 'FoundationBackend'),
            'graph': ('backends.graph.logic', 'GraphBackend')
        }
        
        if backend_name not in backend_map:
            logging.error(f"Unknown backend: {backend_name}")
            sys.exit(1)
        
        module_path, class_name = backend_map[backend_name]
        
        # Import the module and get the class
        module = importlib.import_module(module_path)
        backend_class = getattr(module, class_name)
        
        # Create an instance of the backend
        backend = backend_class()
        
        # Get the appropriate configuration for this backend
        backend_config = getattr(config, f"{backend_name.upper()}_CONFIG", {})
        
        # Initialize the backend with its configuration
        backend.initialize(backend_config)
        
        logging.info(f"Loaded and initialized {backend_name} backend")
        return backend
        
    except Exception as e:
        logging.error(f"Failed to load backend {backend_name}: {str(e)}")
        sys.exit(1)

def main():
    """Main function to run the simulation."""
    start_time = time.time()
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Override config settings with command-line arguments
    backend_name = args.backend
    scenario_id = args.scenario_id
    
    # Set up logging
    log_file = setup_logging(args.log_file, backend_name, scenario_id)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting simulation with backend: {backend_name}")
    logger.info(f"Logging to: {log_file}")
    
    try:
        # Load the backend
        backend = load_backend(backend_name)
        
        # Initialize CV model
        logger.info(f"Initializing CV model from {config.CV_MODEL_PATH} on {config.CV_DEVICE}")
        cv_model = CVModelInterface(config.CV_MODEL_PATH, config.CV_DEVICE)
        
        # Initialize simulation engine
        logger.info("Initializing simulation engine")
        simulation_engine = SimulationEngine(config, backend, cv_model)
        
        # Run the simulation
        if scenario_id:
            logger.info(f"Running scenario: {scenario_id}")
            simulation_engine.run_scenario(scenario_id)
        else:
            logger.info("Running all scenarios")
            simulation_engine.run_all_scenarios()
        
        # Shutdown the backend
        logger.info("Shutting down backend")
        backend.shutdown()
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Simulation completed in {duration:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during simulation: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
