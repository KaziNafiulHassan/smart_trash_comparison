#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base backend interface.
This module defines the common interface that all backends must implement.
"""

import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseBackend(ABC):
    """
    Abstract base class for all backend implementations.

    This class defines the interface that all backend implementations must adhere to.
    It provides abstract methods for initialization, getting sorting suggestions,
    recording interactions, and shutdown.
    """

    def __init__(self):
        """Initialize the backend with a default name."""
        self.name = self.__class__.__name__
        logger.info(f"Creating {self.name} instance")

    @abstractmethod
    def initialize(self, config: dict):
        """
        Initialize the backend with the provided configuration.

        This method should load any necessary resources, establish connections,
        and prepare the backend for use.

        Args:
            config (dict): Configuration parameters for the backend
        """
        pass

    @abstractmethod
    def get_sorting_suggestion(self, item_id: str, user_profile: dict) -> tuple[str, str, float]:
        """
        Get a sorting suggestion for the given item.

        This is the core method that provides waste sorting guidance based on the
        identified item and user profile.

        Args:
            item_id (str): The identified item ID
            user_profile (dict): Information about the user, including preferences,
                                 knowledge level, and history

        Returns:
            tuple[str, str, float]: A tuple containing:
                - suggested_bin (str): The suggested bin for disposal
                - feedback_text (str): Explanatory text for the user
                - processing_latency_ms (float): Processing time in milliseconds
        """
        start_time = time.time()
        # Implement in subclasses
        end_time = time.time()
        processing_latency_ms = (end_time - start_time) * 1000
        return "unknown", "Not implemented", processing_latency_ms

    @abstractmethod
    def record_interaction(self, item_id: str, true_bin: str, suggested_bin: str,
                          chosen_bin: str, user_profile: dict):
        """
        Record details of an interaction for learning and analysis.

        This method should store the interaction details in the backend's specific
        data store if needed, especially for adaptive backends like Graph and Foundation.

        Args:
            item_id (str): The identified item ID
            true_bin (str): The correct bin for the item (ground truth)
            suggested_bin (str): The bin suggested by the backend
            chosen_bin (str): The bin actually chosen by the user
            user_profile (dict): Information about the user
        """
        pass

    @abstractmethod
    def shutdown(self):
        """
        Release resources and perform cleanup before shutdown.

        This method should close connections, save state if necessary,
        and release any resources used by the backend.
        """
        pass

    def log_performance(self, item_id: str, latency_ms: float):
        """
        Log performance metrics for an interaction.

        Args:
            item_id (str): The identified item ID
            latency_ms (float): Processing latency in milliseconds
        """
        logger.info(f"{self.name} processed {item_id} in {latency_ms:.2f}ms")

    def __str__(self):
        """Return a string representation of the backend."""
        return f"{self.name} Backend"

    def __enter__(self):
        """Support for context manager protocol (with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for context manager protocol (with statement)."""
        self.shutdown()
