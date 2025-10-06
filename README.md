# ML/AI Technical Evaluation

## Overview
This is a practical technical assessment for ML/AI engineers. You will work with a transaction dataset and improve a baseline deep learning pipeline.

## Setup
Install dependencies with uv:
```bash
uv sync
```

Run the baseline model:
```bash
uv run train.py
```

## Dataset
- **File**: `data/data.csv`
- **Size**: 10,000 transaction records
- **Features**: 29 columns including transaction details, amounts, dates, and text descriptions
- **Target**: `account_id` (111 unique values)

## Your Task
Improve any part of the deep learning pipeline. This could include:
- Model architecture
- Data preprocessing
- Feature engineering
- Evaluation metrics
- Training strategy
- Task formulation
- Anything else you identify

## Time Allocation
- **15 minutes**: Diagnosis phase - explore the data and baseline model, decide what to improve
- **45-60 minutes**: Implementation phase - make your improvement and document it

## Baseline Model
A basic feedforward neural network is provided with the following structure:
- `dataset.py` - Data loading and feature extraction
- `model.py` - Neural network architecture
- `train.py` - Training script

Features:
- TF-IDF text features (max 100 features)
- Basic numeric features
- Simple 2-layer architecture
- MSE loss with SGD optimizer


## What to Submit
1. Your improved code
2. A brief report (can be markdown, comments, or notebook) explaining:
   - What you diagnosed as the main issue(s)
   - What you changed and why
   - Results/improvements (or theoretical justification if time-limited)
   - Trade-offs of your approach

## Evaluation Criteria
- Problem identification and prioritization
- Technical soundness of the solution
- Ability to explain your choices and trade-offs
- Communication clarity

## Resources
- You have access to an AI assistant (Claude)
- Use any libraries or tools you're comfortable with
- Internet access is available

## Notes
- There's no single "correct" answer - multiple valid improvements exist
- Senior candidates: focus on demonstrating depth and explaining novel approaches
- Junior candidates: focus on clear communication of fundamental improvements
- If you run out of time, explaining your intended approach is valuable

Good luck!
