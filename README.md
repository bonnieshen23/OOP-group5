# Group Project Guide

## Project Overview
- Gymnasium v1.2.2
- Part1 - Traing and testing learning agent
- Part2 - Avoidance of Ice Holes with Expected SARSA
- Part3 - Machine Learning in Ice Hockey and Competing Against Humans
  
## Installation

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
#Windows
.venv\Scripts\activate
#MacOS
source .venv/bin/activate

# 3. Navigate to the Gymnasium directory
cd group_project/Gymnasium

# 4. Install Gymnasium in editable mode
pip install -e .

# 5. Install additional dependencies
pip install "gymnasium[classic_control]"
pip install matplotlib stable-baselines3 pygame pymunk numpy
```
---
## 🚀 Running the Final Project
### **Part 1: Mountain Car**
Train and test the reinforcement learning agent:

```bash
# Train the agent
python mountain_car.py --train --episodes 5000

# Render and visualize performance
python mountain_car.py --render --episodes 10
```

### **Part 2: Frozen Lake**
Run the Frozen Lake environment:

```bash
python frozen_lake.py
```

### **Part 3: OOP Project Environment**
Execute air hockey:

```bash
python air_hockey.py
#For learning mode
1
#For testing and playing mode
2

```
## Dependencies
- gymnasium
- stable-baselines3
- pygame
- pymunk
- numpy

## Contribution list
- part2: 張翊萱
- part3: 鄭心明、沈柏伶
- readme: 沈柏伶
- UML diagrams: 沈柏伶
- reflection: 沈柏伶、鄭心明、張翊萱
- demo slides: 沈柏伶、鄭心明、張翊萱
