#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulation engine for the SMART_TRASH_COMPARISON application.
This module handles the simulation of user interactions with the waste classification system.
"""

import json
import logging
import random
import os
from datetime import datetime

from backends.base_backend import BaseBackend
from core.cv_model import CVModelInterface

logger = logging.getLogger(__name__)

class SimulatedUserModel:
    """
    Model representing a simulated user interacting with the waste classification system.

    This model simulates user behavior based on a profile with initial error rate,
    learning rate, and known biases for specific items.
    """

    def __init__(self, profile: dict):
        """
        Initialize a simulated user model.

        Args:
            profile (dict): User profile parameters including initial_error_rate,
                           learning_rate, and known_bias
        """
        self.initial_error_rate = profile.get('initial_error_rate', 0.3)
        self.learning_rate = profile.get('learning_rate', 0.1)
        self.known_bias = profile.get('known_bias', {})
        self.item_error_probs = {}

        logger.debug(f"Created simulated user model with profile: {profile}")

    def _get_current_error_prob(self, item_id: str) -> float:
        """
        Get the current error probability for an item.

        Args:
            item_id (str): The item ID

        Returns:
            float: The current error probability for the item
        """
        return self.item_error_probs.get(item_id, self.initial_error_rate)

    def choose_bin(self, true_bin: str, suggested_bin: str, feedback_text: str,
                  item_id: str, all_possible_bins: list) -> str:
        """
        Simulate a user choosing a bin based on their profile and the given information.

        Args:
            true_bin (str): The correct bin for the item
            suggested_bin (str): The bin suggested by the backend
            feedback_text (str): The feedback text provided by the backend
            item_id (str): The item ID
            all_possible_bins (list): List of all possible bins

        Returns:
            str: The bin chosen by the simulated user
        """
        # Get current base error probability for this item
        base_error_prob = self._get_current_error_prob(item_id)

        # Check for known bias for this item
        bias_target_bin = None
        bias_boost = 0.0

        if item_id in self.known_bias:
            bias_info = self.known_bias[item_id]
            bias_target_bin = bias_info.get('target_bin')
            bias_boost = bias_info.get('error_prob_boost', 0.0)

        # Generate a random number to determine the outcome
        rand_val = random.random()

        # Decision logic
        if bias_target_bin and rand_val < bias_boost:
            # User chooses the biased bin due to strong bias
            chosen_bin = bias_target_bin
            logger.debug(f"User chose biased bin {bias_target_bin} for item {item_id} (bias_boost={bias_boost})")
        elif rand_val < base_error_prob:
            # User makes an error
            if bias_target_bin and rand_val < (base_error_prob * 0.7):
                # Error is influenced by bias
                chosen_bin = bias_target_bin
                logger.debug(f"User error influenced by bias for item {item_id}")
            else:
                # Random error - choose a random incorrect bin
                incorrect_bins = [bin_name for bin_name in all_possible_bins
                                 if bin_name != true_bin and bin_name != suggested_bin]

                if incorrect_bins:
                    chosen_bin = random.choice(incorrect_bins)
                else:
                    # Fallback if no other bins available
                    chosen_bin = suggested_bin

                logger.debug(f"User made random error for item {item_id}, chose {chosen_bin}")
        else:
            # User chooses correctly
            chosen_bin = true_bin
            logger.debug(f"User chose correctly for item {item_id}: {true_bin}")

        return chosen_bin

    def update_learning(self, item_id: str, was_correct: bool,
                       received_corrective_feedback: bool) -> float:
        """
        Update the user's learning for an item based on the interaction outcome.

        Args:
            item_id (str): The item ID
            was_correct (bool): Whether the user's choice was correct
            received_corrective_feedback (bool): Whether the user received corrective feedback

        Returns:
            float: The updated error probability for the item
        """
        current_error_prob = self._get_current_error_prob(item_id)

        if received_corrective_feedback:
            # Decrease error probability based on learning rate
            new_error_prob = max(0.0, current_error_prob - self.learning_rate)
            self.item_error_probs[item_id] = new_error_prob

            logger.debug(f"User learned from feedback for item {item_id}: "
                        f"error_prob {current_error_prob:.2f} -> {new_error_prob:.2f}")

        return self.item_error_probs.get(item_id, self.initial_error_rate)

    def get_error_prob(self, item_id: str) -> float:
        """
        Get the current error probability for an item.

        Args:
            item_id (str): The item ID

        Returns:
            float: The current error probability for the item
        """
        return self._get_current_error_prob(item_id)


class SimulationEngine:
    """
    Engine for running waste classification simulations.

    This engine simulates user interactions with different backends and
    records the results for analysis.
    """

    def __init__(self, config: dict, backend_instance: BaseBackend,
                cv_model_instance: CVModelInterface):
        """
        Initialize the simulation engine.

        Args:
            config (dict): Configuration parameters
            backend_instance (BaseBackend): The backend implementation to use
            cv_model_instance (CVModelInterface): The CV model implementation to use
        """
        self.config = config
        self.backend = backend_instance
        self.cv_model = cv_model_instance

        # Load scenarios from the specified file
        scenario_file_path = config.get('SCENARIO_FILE_PATH')
        self.scenarios = self._load_scenarios(scenario_file_path)

        # Setup logging
        self._setup_logging()

        logger.info(f"Simulation engine initialized with backend: {type(backend_instance).__name__}")
        logger.info(f"Loaded {len(self.scenarios)} scenarios from {scenario_file_path}")

    def _load_scenarios(self, scenario_file_path: str) -> list:
        """
        Load simulation scenarios from the specified file.

        Args:
            scenario_file_path (str): Path to the scenario file

        Returns:
            list: List of scenario dictionaries
        """
        try:
            with open(scenario_file_path, 'r', encoding='utf-8') as f:
                scenarios = json.load(f)
            return scenarios
        except Exception as e:
            logger.error(f"Failed to load scenarios from {scenario_file_path}: {str(e)}")
            # Return a minimal default scenario
            return [{
                "id": "default_scenario",
                "name": "Default Scenario",
                "description": "A default scenario created when the scenario file could not be loaded",
                "user_profile": {
                    "initial_error_rate": 0.3,
                    "learning_rate": 0.1,
                    "known_bias": {}
                },
                "events": []
            }]

    def _setup_logging(self):
        """Configure logging for the simulation engine."""
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)

        # Create a unique log file name based on timestamp and backend name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backend_name = type(self.backend).__name__
        log_file = os.path.join(log_dir, f"simulation_{backend_name}_{timestamp}.log")

        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        # Add the handler to the logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        # Set the overall level to DEBUG
        root_logger.setLevel(logging.DEBUG)

        logger.info(f"Logging configured. Log file: {log_file}")

    def run_scenario(self, scenario_id: str):
        """
        Run a specific simulation scenario.

        Args:
            scenario_id (str): The ID of the scenario to run

        Returns:
            dict: Results of the simulation
        """
        # Find the scenario by ID
        scenario = next((s for s in self.scenarios if s.get('id') == scenario_id), None)

        if not scenario:
            error_msg = f"Scenario with ID '{scenario_id}' not found"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Starting scenario: {scenario.get('name', scenario_id)}")
        logger.info(f"Description: {scenario.get('description', 'No description')}")

        # Create a simulated user model based on the scenario's user profile
        user_profile = scenario.get('user_profile', {})
        self.user_model = SimulatedUserModel(user_profile)

        # Load rules for ground truth
        try:
            rule_file_path = self.config.get('RULE_FILE_PATH')
            with open(rule_file_path, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
            logger.info(f"Loaded {len(self.rules)} rules from {rule_file_path}")
        except Exception as e:
            logger.error(f"Failed to load rules from {rule_file_path}: {str(e)}")
            self.rules = []

        # Get the events for this scenario
        events = scenario.get('events', [])
        logger.info(f"Scenario has {len(events)} events")

        # Collect results for each event
        results = []

        # Process each event
        for i, event in enumerate(events):
            event_type = event.get('event_type', 'unknown')
            item_id = event.get('item_id', 'unknown_item')
            event_id = event.get('id', f"event_{i}")

            logger.info(f"Processing event {i+1}/{len(events)}: {event_type} for item {item_id}")

            # Get ground truth for the item
            true_bin = "Unknown"
            for rule in self.rules:
                if rule.get('ItemID') == item_id:
                    true_bin = rule.get('CorrectBin_Magdeburg', 'Unknown')
                    break

            # Common user profile for all backend calls
            user_profile_dict = {'user_id': 'sim_user'}

            # Event result dictionary to store metrics
            event_result = {
                'timestamp': datetime.now().isoformat(),
                'event_id': event_id,
                'backend_type': type(self.backend).__name__,
                'scenario_id': scenario_id,
                'user_profile_type': user_profile.get('type', 'default'),
                'event_type': event_type,
                'item_id': item_id,
                'true_bin': true_bin
            }

            try:
                # Handle different event types
                if event_type == 'identify':
                    # CV model prediction
                    predicted_item_id = item_id  # Default if CV fails
                    confidence = 0.0

                    try:
                        if self.cv_model and 'image_path' in event:
                            cv_result = self.cv_model.predict(event['image_path'])
                            if cv_result:
                                predicted_item_id = cv_result.get('item_id', item_id)
                                confidence = cv_result.get('confidence', 0.0)
                                logger.debug(f"CV model predicted {predicted_item_id} with confidence {confidence:.2f}")
                    except Exception as e:
                        logger.error(f"Error in CV model prediction: {str(e)}")

                    # Backend suggestion
                    suggested_bin = "Unknown"
                    feedback_text = "Error getting suggestion"
                    latency = 0.0

                    try:
                        backend_result = self.backend.get_sorting_suggestion(
                            predicted_item_id,
                            user_profile=user_profile_dict
                        )

                        if backend_result and len(backend_result) >= 3:
                            suggested_bin, feedback_text, latency = backend_result
                            logger.debug(f"Backend suggested {suggested_bin} with latency {latency:.2f}ms")
                    except Exception as e:
                        logger.error(f"Error getting backend suggestion: {str(e)}")

                    # Update event result with identify-specific metrics
                    event_result.update({
                        'predicted_item_id': predicted_item_id,
                        'cv_confidence': confidence,
                        'suggested_bin': suggested_bin,
                        'feedback_text': feedback_text,
                        'processing_latency_ms': latency,
                        'chosen_bin': None,
                        'is_correct': (suggested_bin == true_bin),
                        'user_error_prob_before': None,
                        'user_error_prob_after': None
                    })

                elif event_type == 'game_sort':
                    # Get user's current error probability for this item
                    error_prob_before = self.user_model.get_error_prob(item_id)

                    # Backend suggestion
                    suggested_bin = "Unknown"
                    feedback_text = "Error getting suggestion"
                    latency = 0.0

                    try:
                        backend_result = self.backend.get_sorting_suggestion(
                            item_id,
                            user_profile=user_profile_dict
                        )

                        if backend_result and len(backend_result) >= 3:
                            suggested_bin, feedback_text, latency = backend_result
                            logger.debug(f"Backend suggested {suggested_bin} with latency {latency:.2f}ms")
                    except Exception as e:
                        logger.error(f"Error getting backend suggestion: {str(e)}")

                    # Get all possible bins from rules
                    all_bins = []
                    for rule in self.rules:
                        bin_name = rule.get('CorrectBin_Magdeburg')
                        if bin_name and bin_name not in all_bins:
                            all_bins.append(bin_name)

                    if not all_bins:
                        all_bins = ["Paper", "Plastic", "Glass", "Metal", "Organic", "Residual"]

                    # Simulate user choice
                    chosen_bin = self.user_model.choose_bin(
                        true_bin,
                        suggested_bin,
                        feedback_text,
                        item_id,
                        all_bins
                    )

                    # Determine if the choice was correct
                    is_correct = (chosen_bin == true_bin)

                    # Determine if corrective feedback was received
                    received_corrective = (not is_correct and suggested_bin == true_bin)

                    # Update user learning
                    error_prob_after = self.user_model.update_learning(
                        item_id,
                        is_correct,
                        received_corrective
                    )

                    # Record interaction in backend
                    try:
                        self.backend.record_interaction(
                            item_id,
                            true_bin,
                            suggested_bin,
                            chosen_bin,
                            user_profile=user_profile_dict
                        )
                    except Exception as e:
                        logger.error(f"Error recording interaction in backend: {str(e)}")

                    # Update event result with game_sort-specific metrics
                    event_result.update({
                        'cv_prediction': None,
                        'cv_confidence': None,
                        'suggested_bin': suggested_bin,
                        'feedback_text': feedback_text,
                        'processing_latency_ms': latency,
                        'chosen_bin': chosen_bin,
                        'is_correct': is_correct,
                        'user_error_prob_before': error_prob_before,
                        'user_error_prob_after': error_prob_after
                    })

                else:
                    logger.warning(f"Unknown event type: {event_type}")
                    event_result['status'] = 'skipped'

                # Add event result to results list
                results.append(event_result)

                # Log the event result
                log_msg = f"Event {event_id} ({event_type}): "
                if event_type == 'identify':
                    log_msg += f"CV predicted {predicted_item_id} ({confidence:.2f}), "
                    log_msg += f"Backend suggested {suggested_bin}, "
                    log_msg += f"Correct: {event_result['is_correct']}"
                elif event_type == 'game_sort':
                    log_msg += f"User chose {chosen_bin}, "
                    log_msg += f"Correct: {is_correct}, "
                    log_msg += f"Error prob: {error_prob_before:.2f} -> {error_prob_after:.2f}"

                logger.info(log_msg)

            except Exception as e:
                logger.error(f"Error processing event {event_id}: {str(e)}")
                event_result['status'] = 'error'
                event_result['error_message'] = str(e)
                results.append(event_result)

        logger.info(f"Completed scenario: {scenario.get('name', scenario_id)}")

        # Save results to a log file
        self._save_scenario_results(scenario_id, results)

        return {
            "scenario_id": scenario_id,
            "status": "completed",
            "events_processed": len(results),
            "results": results
        }

    def _save_scenario_results(self, scenario_id: str, results: list):
        """
        Save scenario results to a log file.

        Args:
            scenario_id (str): The ID of the scenario
            results (list): List of event results
        """
        try:
            # Create logs directory if it doesn't exist
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
            os.makedirs(log_dir, exist_ok=True)

            # Create a unique log file name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backend_name = type(self.backend).__name__
            log_file = os.path.join(log_dir, f"{backend_name}_{scenario_id}_{timestamp}.json")

            # Save results as JSON
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

            logger.info(f"Saved scenario results to {log_file}")

        except Exception as e:
            logger.error(f"Error saving scenario results: {str(e)}")
