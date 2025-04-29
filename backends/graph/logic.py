#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Graph backend implementation.
This module implements a backend using Neo4j graph database for waste classification.
"""

import logging
import time
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from backends.base_backend import BaseBackend

logger = logging.getLogger(__name__)

class GraphBackend(BaseBackend):
    """
    Graph backend implementation using Neo4j.

    This backend uses a Neo4j graph database to provide personalized, adaptive
    feedback for waste sorting suggestions based on user interactions and
    knowledge graph relationships.
    """

    def __init__(self):
        """Initialize the graph backend."""
        super().__init__()
        self.driver = None
        logger.info("Graph backend instance created")

    def initialize(self, config: dict):
        """
        Initialize the graph backend with the provided configuration.

        Args:
            config (dict): Configuration parameters including Neo4j connection details
        """
        logger.info("Initializing Graph backend")

        # Read Neo4j connection details from config
        neo4j_config = config.get('NEO4J_CONFIG')
        if not neo4j_config:
            logger.error("No Neo4j configuration found in config")
            raise ValueError("NEO4J_CONFIG must be specified in config")

        uri = neo4j_config.get('uri')
        user = neo4j_config.get('user')
        password = neo4j_config.get('password')

        if not uri or not user or not password:
            logger.error("Missing Neo4j connection details (uri, user, or password)")
            raise ValueError("Neo4j connection details (uri, user, password) must be specified in config")

        # Create the Neo4j driver instance
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))

            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database")

        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j database: {str(e)}")
            raise ConnectionError(f"Could not connect to Neo4j database at {uri}: {str(e)}")
        except AuthError as e:
            logger.error(f"Authentication failed for Neo4j database: {str(e)}")
            raise ConnectionError(f"Authentication failed for Neo4j database: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j database: {str(e)}")
            raise

    def _execute_read_query(self, query, parameters=None):
        """
        Execute a read-only Cypher query.

        Args:
            query (str): The Cypher query to execute
            parameters (dict, optional): Parameters for the query

        Returns:
            list: Results of the query
        """
        if not self.driver:
            logger.warning("No Neo4j connection available")
            return []

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing read query: {str(e)}")
            return []

    def _execute_write_query(self, query, parameters=None):
        """
        Execute a write Cypher query within a transaction.

        Args:
            query (str): The Cypher query to execute
            parameters (dict, optional): Parameters for the query

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.driver:
            logger.warning("No Neo4j connection available")
            return False

        try:
            with self.driver.session() as session:
                session.execute_write(
                    lambda tx: tx.run(query, parameters or {})
                )
            return True
        except Exception as e:
            logger.error(f"Error executing write query: {str(e)}")
            return False

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

        # Default values in case of errors
        correct_bin = "Unknown"
        feedback = "No information available for this item."

        try:
            # Get user ID from user profile
            user_id = user_profile.get('user_id', 'unknown_user')

            # Query 1: Find the correct bin for the item_id and 'Magdeburg'
            bin_query = """
            MATCH (i:WasteItem {id: $item_id})-[:SHOULD_BE_IN]->(b:Bin)
            WHERE EXISTS((i)-[:APPLIES_TO]->(:Region {name: 'Magdeburg'}))
            RETURN b.name AS bin_name
            """

            bin_results = self._execute_read_query(bin_query, {'item_id': item_id})

            if bin_results and 'bin_name' in bin_results[0]:
                correct_bin = bin_results[0]['bin_name']
                logger.debug(f"Found correct bin for {item_id}: {correct_bin}")
            else:
                logger.warning(f"No bin found for item {item_id} in Magdeburg")
                correct_bin = "Unknown"

            # Query 2: Find recent errors for this user and item
            errors_query = """
            MATCH (u:User {id: $user_id})-[e:MADE_ERROR]->(i:WasteItem {id: $item_id})
            RETURN e.chosen_bin AS wrong_bin, e.timestamp AS timestamp
            ORDER BY e.timestamp DESC
            LIMIT 3
            """

            error_results = self._execute_read_query(errors_query, {
                'user_id': user_id,
                'item_id': item_id
            })

            recent_errors = []
            if error_results:
                for error in error_results:
                    if 'wrong_bin' in error:
                        recent_errors.append(error['wrong_bin'])
                logger.debug(f"Found recent errors for user {user_id} and item {item_id}: {recent_errors}")

            # Query 3: Fetch a relevant feedback snippet
            snippet_query = """
            MATCH (i:WasteItem {id: $item_id})-[:HAS_FEEDBACK]->(f:FeedbackSnippet)
            WHERE f.language = $language
            RETURN f.text AS snippet_text
            LIMIT 1
            """

            snippet_results = self._execute_read_query(snippet_query, {
                'item_id': item_id,
                'language': 'de'  # Default language
            })

            snippet_text = ""
            if snippet_results and 'snippet_text' in snippet_results[0]:
                snippet_text = snippet_results[0]['snippet_text']
                logger.debug(f"Found feedback snippet for {item_id}")

            # Get item name from the database or use item_id if not found
            item_name_query = """
            MATCH (i:WasteItem {id: $item_id})
            RETURN i.name_de AS item_name
            """

            item_name_results = self._execute_read_query(item_name_query, {'item_id': item_id})
            item_name = item_id
            if item_name_results and 'item_name' in item_name_results[0]:
                item_name = item_name_results[0]['item_name']

            # Implement adaptive feedback selection logic
            if recent_errors:
                # Prioritize feedback addressing the most recent specific mistake
                most_recent_wrong_bin = recent_errors[0]
                feedback = f"Remember, {item_name} goes in the {correct_bin}, not the {most_recent_wrong_bin} like last time!"
            elif snippet_text:
                # Use the relevant snippet if available
                feedback = snippet_text
            else:
                # Construct a standard positive feedback string
                feedback = f"{item_name} belongs in the {correct_bin}."

        except Exception as e:
            logger.error(f"Error in get_sorting_suggestion for {item_id}: {str(e)}")
            correct_bin = "Unknown"
            feedback = f"An error occurred while processing your request: {str(e)}"

        # Calculate latency
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        return (correct_bin, feedback, latency_ms)

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
        # Get user ID from user profile
        user_id = user_profile.get('user_id', 'unknown_user')

        # Log the interaction
        logger.info(f"Recording interaction for user {user_id}, item {item_id}: true={true_bin}, suggested={suggested_bin}, chosen={chosen_bin}")

        # Check if an error occurred (chosen bin doesn't match true bin)
        if chosen_bin != true_bin:
            # Create an error relationship in the graph
            error_query = """
            MATCH (u:User {id: $user_id})
            MATCH (i:WasteItem {id: $item_id})
            CREATE (u)-[e:MADE_ERROR]->(i)
            SET e.chosen_bin = $chosen_bin,
                e.true_bin = $true_bin,
                e.suggested_bin = $suggested_bin,
                e.timestamp = datetime()
            """

            # If the user or item doesn't exist, create them
            create_missing_nodes_query = """
            MERGE (u:User {id: $user_id})
            MERGE (i:WasteItem {id: $item_id})
            """

            # Execute the queries
            self._execute_write_query(create_missing_nodes_query, {
                'user_id': user_id,
                'item_id': item_id
            })

            success = self._execute_write_query(error_query, {
                'user_id': user_id,
                'item_id': item_id,
                'chosen_bin': chosen_bin,
                'true_bin': true_bin,
                'suggested_bin': suggested_bin
            })

            if success:
                logger.info(f"Recorded error for user {user_id}, item {item_id}: chose {chosen_bin} instead of {true_bin}")
            else:
                logger.error(f"Failed to record error for user {user_id}, item {item_id}")
        else:
            # Record a successful interaction
            success_query = """
            MATCH (u:User {id: $user_id})
            MATCH (i:WasteItem {id: $item_id})
            CREATE (u)-[s:SORTED_CORRECTLY]->(i)
            SET s.bin = $true_bin,
                s.timestamp = datetime()
            """

            # If the user or item doesn't exist, create them
            create_missing_nodes_query = """
            MERGE (u:User {id: $user_id})
            MERGE (i:WasteItem {id: $item_id})
            """

            # Execute the queries
            self._execute_write_query(create_missing_nodes_query, {
                'user_id': user_id,
                'item_id': item_id
            })

            success = self._execute_write_query(success_query, {
                'user_id': user_id,
                'item_id': item_id,
                'true_bin': true_bin
            })

            if success:
                logger.info(f"Recorded successful sorting for user {user_id}, item {item_id}: {true_bin}")
            else:
                logger.error(f"Failed to record successful sorting for user {user_id}, item {item_id}")

    def shutdown(self):
        """
        Release resources and perform cleanup before shutdown.

        Closes the Neo4j driver connection if it exists.
        """
        if self.driver:
            try:
                self.driver.close()
                logger.info("Neo4j driver connection closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j driver connection: {str(e)}")
                # Continue with shutdown even if there's an error
