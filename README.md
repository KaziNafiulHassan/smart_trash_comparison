# SMART_TRASH_COMPARISON

A comprehensive framework for comparing different waste sorting assistance backends, including traditional rule-based systems, foundation models (LLMs/VLMs), and graph-based knowledge systems.

## Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Backends](#backends)
  - [Baseline Backend](#baseline-backend)
  - [Foundation Backend](#foundation-backend)
  - [Graph Backend](#graph-backend)
- [Simulation Framework](#simulation-framework)
- [Analysis Tools](#analysis-tools)
- [Data Files](#data-files)
- [Development](#development)
- [License](#license)

## Overview

The SMART_TRASH_COMPARISON project provides a framework for comparing different approaches to waste sorting assistance systems. It includes:

1. A modular architecture with swappable backends
2. A core CV model for item identification
3. Multiple backend implementations:
   - Baseline (rule-based)
   - Foundation (LLM/VLM-based)
   - Graph (Neo4j knowledge graph-based)
4. A simulation framework for testing backends with simulated users
5. Analysis tools for evaluating and comparing backend performance

This project allows researchers and developers to:
- Compare different AI approaches for waste sorting assistance
- Simulate user interactions with different levels of knowledge and biases
- Analyze the effectiveness of different feedback mechanisms
- Evaluate the performance of different backends in terms of accuracy and latency

## Project Architecture

The project follows a modular architecture with the following components:

```
SMART_TRASH_COMPARISON/
├── backends/                  # Backend implementations
│   ├── base_backend.py        # Abstract base class for all backends
│   ├── baseline/              # Rule-based baseline backend
│   │   └── logic.py           # Implementation of baseline backend
│   ├── foundation/            # LLM/VLM-based backend
│   │   └── logic.py           # Implementation of foundation backend
│   └── graph/                 # Neo4j graph-based backend
│       └── logic.py           # Implementation of graph backend
├── core/                      # Core functionality
│   ├── cv_model.py            # Computer vision model interface
│   └── simulation.py          # Simulation engine
├── analysis/                  # Analysis tools
│   ├── parse_logs.py          # Tool for parsing simulation logs
│   └── generate_plots.py      # Tool for generating comparison plots
├── data/                      # Data files
│   ├── magdeburg_rules.json   # Waste sorting rules for Magdeburg
│   └── simulation_scenarios.json # Simulation scenarios
├── logs/                      # Log files (generated during runtime)
├── plots/                     # Generated plots (output of analysis)
├── config.py                  # Configuration settings
└── main.py                    # Main entry point
```

### Component Interactions

The system components interact as follows:

1. The `app.py` script initializes the system based on configuration in `config.py`
2. The CV model identifies waste items from images
3. The selected backend provides sorting suggestions based on the identified items
4. The simulation engine simulates user interactions with the backend
5. Logs are generated during simulation
6. Analysis tools parse logs and generate comparison plots

## Installation

### Prerequisites

- Python 3.8 or higher
- Neo4j database (for the Graph backend)
- API keys for OpenAI, Google, or Anthropic (for the Foundation backend)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SMART_TRASH_COMPARISON.git
   cd SMART_TRASH_COMPARISON
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables for API keys (for Foundation backend):
   ```bash
   # For OpenAI
   export OPENAI_API_KEY=your_api_key_here
   
   # For Google
   export GOOGLE_API_KEY=your_api_key_here
   
   # For Anthropic
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

5. Set up Neo4j (for Graph backend):
   - Install Neo4j (https://neo4j.com/download/)
   - Create a new database
   - Update the Neo4j connection details in `config.py`

## Configuration

The system is configured through the `config.py` file. Key configuration parameters include:

### General Configuration

```python
# File paths
RULE_FILE_PATH = 'data/magdeburg_rules.json'
SCENARIO_FILE_PATH = 'data/simulation_scenarios.json'
LOG_DIR = 'logs/'
PLOT_DIR = 'plots/'

# Backend selection
ACTIVE_BACKEND = 'foundation'  # Options: 'baseline', 'foundation', 'graph'
```

### Foundation Backend Configuration

```python
FOUNDATION_MODEL_CONFIG = {
    'model_type': 'openai_gpt4',  # Options: 'openai_gpt4', 'openai_gpt35', 'google_gemini', 'anthropic_claude'
    'api_key_env_var': 'OPENAI_API_KEY',
    'endpoint': 'https://api.openai.com/v1/chat/completions',
    'model_name': 'gpt-4'
}
```

### Graph Backend Configuration

```python
NEO4J_CONFIG = {
    'uri': 'bolt://localhost:7687',
    'user': 'neo4j',
    'password': 'password'
}
```

### Simulation Configuration

```python
SIMULATION_CONFIG = {
    'num_runs': 5,
    'log_level': 'INFO'
}
```

## Usage

### Running a Simulation

To run a simulation with the configured backend:

```bash
python app.py --scenario standard_user
```

Options:
- `--scenario`: ID of the scenario to run (default: all scenarios)
- `--backend`: Override the backend specified in config.py
- `--runs`: Number of simulation runs (overrides config)
- `--verbose`: Enable verbose logging

### Analyzing Results

1. Parse the simulation logs:

```bash
python analysis/parse_logs.py --log_dir logs/ --output combined_data.pkl
```

2. Generate comparison plots:

```bash
python analysis/generate_plots.py --data_file combined_data.pkl --output_dir plots/
```

## Backends

The system includes three backend implementations, each with different approaches to waste sorting assistance.

### Baseline Backend

The Baseline backend is a simple rule-based system that uses direct lookups in the Magdeburg rules database to provide sorting suggestions. It serves as a baseline for comparing more advanced approaches.

**Key features:**
- Fast, deterministic responses
- No learning or adaptation
- Simple implementation
- No external dependencies

**Implementation details:**
- Located in `backends/baseline/logic.py`
- Implements the `BaseBackend` interface
- Uses direct lookups in the rules database

### Foundation Backend

The Foundation backend uses large language models (LLMs) or vision-language models (VLMs) to generate personalized feedback for waste sorting. It combines rule-based knowledge with the natural language capabilities of foundation models.

**Key features:**
- Natural language feedback
- Personalization based on user profile
- Explanations and educational content
- Support for multiple LLM/VLM providers

**Implementation details:**
- Located in `backends/foundation/logic.py`
- Implements the `BaseBackend` interface
- Supports OpenAI, Google, and Anthropic models
- Uses prompt engineering to generate helpful feedback

**Configuration options:**
- `model_type`: The type of model to use (e.g., 'openai_gpt4')
- `api_key_env_var`: Environment variable containing the API key
- `endpoint`: API endpoint URL
- `model_name`: Specific model name

### Graph Backend

The Graph backend uses a Neo4j knowledge graph to provide adaptive feedback based on user interactions and knowledge relationships. It builds a graph of waste items, bins, users, and their interactions to provide personalized suggestions.

**Key features:**
- Adaptive feedback based on user history
- Knowledge graph of waste sorting relationships
- Learning from user interactions
- Network analysis capabilities

**Implementation details:**
- Located in `backends/graph/logic.py`
- Implements the `BaseBackend` interface
- Uses Neo4j for graph storage and queries
- Implements Cypher queries for suggestion and interaction recording

**Configuration options:**
- `uri`: Neo4j database URI
- `user`: Neo4j username
- `password`: Neo4j password

## Simulation Framework

The simulation framework allows testing backends with simulated users of varying knowledge levels and biases.

### Simulated User Model

The `SimulatedUserModel` class simulates user behavior with the following parameters:

- `initial_error_rate`: Probability of making an error initially
- `learning_rate`: How quickly the user learns from feedback
- `known_bias`: Specific biases for certain items

### Simulation Scenarios

Scenarios are defined in `data/simulation_scenarios.json` and include:

- User profiles with different knowledge levels and biases
- Sequences of events (identify, game_sort)
- Item IDs and image paths

Example scenario:
```json
{
  "id": "beginner_user",
  "name": "Beginner User Scenario",
  "description": "Simulates a beginner user with high error rate and slow learning",
  "user_profile": {
    "initial_error_rate": 0.5,
    "learning_rate": 0.05,
    "known_bias": {
      "item123": {
        "target_bin": "Residual",
        "error_prob_boost": 0.3
      }
    }
  },
  "events": [
    {
      "id": "event1",
      "event_type": "identify",
      "item_id": "item123",
      "image_path": "data/images/item123.jpg"
    },
    {
      "id": "event2",
      "event_type": "game_sort",
      "item_id": "item123"
    }
  ]
}
```

### Event Types

The simulation supports two types of events:

1. **identify**: Simulates the CV model identifying an item from an image
2. **game_sort**: Simulates the user choosing a bin for an identified item

## Analysis Tools

The project includes tools for analyzing simulation results and comparing backend performance.

### Log Parsing

The `parse_logs.py` script converts simulation logs into a structured DataFrame for analysis:

```bash
python analysis/parse_logs.py --log_dir logs/ --output combined_data.pkl
```

### Plot Generation

The `generate_plots.py` script generates comparison plots:

```bash
python analysis/generate_plots.py --data_file combined_data.pkl --output_dir plots/
```

Generated plots include:

1. **Overall Accuracy**: Bar plot comparing accuracy across backends
2. **Latency**: Box plots and violin plots of processing latency
3. **Accuracy Over Time**: Line plot showing learning effects
4. **Error Analysis**: Heatmaps and bar charts of error patterns

## Data Files

### Magdeburg Rules

The `magdeburg_rules.json` file contains waste sorting rules for Magdeburg, Germany. Each entry includes:

- `ItemID`: Unique identifier for the item
- `ItemName_DE`: German name of the item
- `CorrectBin_Magdeburg`: The correct bin for disposal
- `Notes_DE`: Additional notes in German

Example:
```json
[
  {
    "ItemID": "plastic_bottle",
    "ItemName_DE": "Plastikflasche",
    "CorrectBin_Magdeburg": "Yellow Bin",
    "Notes_DE": "Bitte vorher leeren und zusammendrücken"
  }
]
```

### Simulation Scenarios

The `simulation_scenarios.json` file defines scenarios for testing backends with different user profiles and event sequences. See the [Simulation Scenarios](#simulation-scenarios) section for details.

## Development

### Adding a New Backend

To add a new backend:

1. Create a new directory in `backends/`
2. Implement a class that inherits from `BaseBackend` in `backends/base_backend.py`
3. Implement the required methods:
   - `initialize(self, config: dict)`
   - `get_sorting_suggestion(self, item_id: str, user_profile: dict)`
   - `record_interaction(self, item_id: str, true_bin: str, suggested_bin: str, chosen_bin: str, user_profile: dict)`
   - `shutdown(self)`
4. Update `config.py` to include configuration for your backend
5. Update `main.py` to support loading your backend

### Testing

Run tests with:

```bash
python -m unittest discover tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Citation

If you use this project in your research, please cite:

```
@software{smart_trash_comparison,
  author = {Your Name},
  title = {SMART_TRASH_COMPARISON: A Framework for Comparing Waste Sorting Assistance Backends},
  year = {2023},
  url = {https://github.com/yourusername/SMART_TRASH_COMPARISON}
}
```
