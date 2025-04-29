#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Abstract interface for UI updates.
This module provides a common interface for updating the UI across different backends.
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class UIInterface(ABC):
    """Abstract base class for UI interfaces."""
    
    @abstractmethod
    def update_status(self, status_message):
        """
        Update the status message in the UI.
        
        Args:
            status_message (str): Status message to display
        """
        pass
    
    @abstractmethod
    def display_result(self, result):
        """
        Display a classification result in the UI.
        
        Args:
            result (dict): Classification result to display
        """
        pass
    
    @abstractmethod
    def show_error(self, error_message):
        """
        Display an error message in the UI.
        
        Args:
            error_message (str): Error message to display
        """
        pass
    
    @abstractmethod
    def request_user_input(self, prompt, options=None):
        """
        Request input from the user.
        
        Args:
            prompt (str): Prompt to display to the user
            options (list, optional): List of options for the user to choose from
            
        Returns:
            str: User input
        """
        pass


class ConsoleUI(UIInterface):
    """Console-based implementation of the UI interface."""
    
    def update_status(self, status_message):
        """
        Update the status message in the console.
        
        Args:
            status_message (str): Status message to display
        """
        print(f"STATUS: {status_message}")
        logger.info(f"UI Status: {status_message}")
    
    def display_result(self, result):
        """
        Display a classification result in the console.
        
        Args:
            result (dict): Classification result to display
        """
        print("\n=== Classification Result ===")
        print(f"Class: {result.get('class', 'Unknown')}")
        print(f"Confidence: {result.get('confidence', 0):.2%}")
        
        if 'recommendation' in result:
            print(f"\nRecommendation: {result['recommendation']}")
        
        if 'all_probabilities' in result:
            print("\nAll Probabilities:")
            for cls, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {cls}: {prob:.2%}")
        
        print("============================\n")
        logger.debug(f"Displayed result: {result}")
    
    def show_error(self, error_message):
        """
        Display an error message in the console.
        
        Args:
            error_message (str): Error message to display
        """
        print(f"ERROR: {error_message}")
        logger.error(f"UI Error: {error_message}")
    
    def request_user_input(self, prompt, options=None):
        """
        Request input from the user via the console.
        
        Args:
            prompt (str): Prompt to display to the user
            options (list, optional): List of options for the user to choose from
            
        Returns:
            str: User input
        """
        if options:
            option_str = ", ".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
            full_prompt = f"{prompt}\nOptions: {option_str}\nEnter your choice (1-{len(options)}): "
        else:
            full_prompt = f"{prompt}: "
        
        user_input = input(full_prompt)
        logger.debug(f"User input: {user_input}")
        
        if options:
            try:
                choice = int(user_input)
                if 1 <= choice <= len(options):
                    return options[choice - 1]
                else:
                    print("Invalid choice. Please try again.")
                    return self.request_user_input(prompt, options)
            except ValueError:
                print("Invalid input. Please enter a number.")
                return self.request_user_input(prompt, options)
        
        return user_input


class WebUI(UIInterface):
    """Web-based implementation of the UI interface using Flask."""
    
    def __init__(self, app=None):
        """
        Initialize the web UI.
        
        Args:
            app: Flask application instance
        """
        self.app = app
        self.socket = None  # For real-time updates with Flask-SocketIO
        logger.info("Web UI initialized")
    
    def update_status(self, status_message):
        """
        Update the status message in the web UI.
        
        Args:
            status_message (str): Status message to display
        """
        if self.socket:
            self.socket.emit('status_update', {'message': status_message})
        logger.info(f"Web UI Status: {status_message}")
    
    def display_result(self, result):
        """
        Display a classification result in the web UI.
        
        Args:
            result (dict): Classification result to display
        """
        if self.socket:
            self.socket.emit('classification_result', result)
        logger.debug(f"Web UI displayed result: {result}")
    
    def show_error(self, error_message):
        """
        Display an error message in the web UI.
        
        Args:
            error_message (str): Error message to display
        """
        if self.socket:
            self.socket.emit('error', {'message': error_message})
        logger.error(f"Web UI Error: {error_message}")
    
    def request_user_input(self, prompt, options=None):
        """
        Request input from the user via the web UI.
        
        Args:
            prompt (str): Prompt to display to the user
            options (list, optional): List of options for the user to choose from
            
        Returns:
            str: User input
        """
        # This would typically be implemented with a form or modal in the web UI
        # For now, we'll just log that this was requested
        logger.info(f"Web UI requested user input: {prompt}, options: {options}")
        return None  # In a real implementation, this would wait for user input
