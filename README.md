# AE4350 Hummingbird Reinforcement Learning Project

## Course Project Overview

This repository contains the implementation of a reinforcement learning project for the AE4350 course, focusing on training an autonomous hummingbird agent to efficiently collect nectar in a complex 3D environment. The project demonstrates advanced concepts in reinforcement learning, including PPO (Proximal Policy Optimization) algorithm implementation, environment design, and comprehensive performance analysis.

## Features

### 3D Hummingbird Environment
- **3D Navigation**: Hummingbird moves in x, y, z coordinates in physics informed environment 
- **Energy Management System**: Agent must manage energy consumption for movement, hovering, and metabolic costs
- **Multiple Flowers**: Environment contains multiple nectar sources at different heights with regeneration mechanics 
- **Flower Cooldowns**: Flowers have cooldown periods after being visited to prevent camping

### Reinforcement Learning Implementation
- **PPO Algorithm**: Uses Stable Baselines3 PPO implementation for policy optimization
- **Autonomous Learning**: Minimal reward engineering to allow emergent strategies
- **Real-time Visualization**: Matplotlib-based 3D visualization during training and testing

### Analysis and Evaluation
- **Performance Metrics**: Tracks rewards, episode lengths, survival rates, nectar collection, and altitude statistics
- **Statistical Analysis**: Detailed evaluation with multiple runs and statistical significance testing
- **Visualization Tools**: Multiple plotting scripts for action distributions, trajectories, and performance comparisons
- **Model Comparison**: Tools to compare different trained models and analyze their strategies

### Project Tools
- **Interactive Launcher**: menu system for all project operations
- **Model Management**: Organized storage and versioning of trained models
- **Debugging Tools**: Various debugging and testing utilities for environment and model validation

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Dependencies
The project requires the following Python packages:
- `gymnasium` - Environment framework
- `stable-baselines3` - Reinforcement learning algorithms
- `torch` - PyTorch for neural networks
- `numpy` - Numerical computations
- `matplotlib` - 3D visualization
- `scipy` - Statistical analysis

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd AE4350
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install gymnasium stable-baselines3 torch numpy matplotlib scipy
```

## Usage

### Quick Start
Run the launcher script to access all project features:
```bash
python launcher.py
```

### Available Operations
1. **Test Environment** - Watch the 3D hummingbird in action
2. **Train New Model** - Start training with predefined timesteps (500K or 1M)
3. **Custom Training** - Specify exact number of training timesteps
4. **Continue Training** - Resume training from existing model
5. **Test Trained Model** - Evaluate trained agent performance
6. **Evaluate Performance** - Run detailed statistical evaluation
7. **View Progress** - Visualize training statistics and learning curves
8. **Environment Analysis** - Analyze environment difficulty and dynamics
9. **Model Comparison** - Compare performance of different models
10. **Compatibility Check** - Verify model compatibility with current environment
11. **Detailed Evaluation** - Comprehensive model assessment

### Training a Model
```bash
# From launcher menu, choose option 2 or 3
# Or run directly:
python train.py
```

### Testing a Model
```bash
# From launcher menu, choose option 6
# Or run directly:
python detailed_evaluation.py <model_path>
```

## Project Structure

```
AE4350/
├── train.py                    # Main training script with PPO
├── hummingbird_env.py          # 3D hummingbird environment implementation
├── launcher.py                 # Interactive menu system
├── detailed_evaluation.py      # Comprehensive model evaluation
├── Figures/                    # Generated figures and reports
│   ├── environment_snapshot.pdf
│   └── *.png                   # Various analysis plots
├── models/                     # Trained model files
│   ├── *.zip                   # Saved PPO models
│   ├── *_training_stats.pkl    # Training statistics
│   └── *.png                   # Training progress plots
├── scripts/                    # Analysis and utility scripts
│   ├── *.py                    # Various analysis tools
│   └── analysis/               # Advanced analysis scripts
│       ├── scripts/            # Detailed analysis utilities
│       └── outputs/            # Analysis results
└── README.md                   # This file
```

## Main Components

### Environment (`hummingbird_env.py`)
- Custom Gym environment with 3D matplotlib visualization
- Implements simple hummingbird flight physics
- Manages flower states, energy consumption, and reward calculation

### Training (`train.py`)
- PPO training loop with logging
- Callback for tracking detailed statistics
- Support for resuming training from checkpoints

### Evaluation (`detailed_evaluation.py`)
- Statistical evaluation of trained models
- Multiple run analysis for robust performance assessment
- Comparison of different training configurations

## Results and Analysis

The project includes comprehensive analysis tools to evaluate:
- **Learning Progress**: Reward curves, episode lengths, survival rates
- **Strategy Analysis**: Action distributions, trajectory patterns
- **Performance Metrics**: Nectar collection efficiency, energy management
- **Model Comparisons**: Statistical significance testing between models

Generated figures include:
- Training reward progression
- Episode length distributions
- Survival rate analysis
- Nectar collection statistics
- Altitude management plots
- Action histograms
- Trajectory visualizations

---
