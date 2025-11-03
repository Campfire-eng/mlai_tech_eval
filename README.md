# ML/AI Technical Evaluation

## Setup

### 1. Install uv (if not already installed)

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or see [uv installation docs](https://github.com/astral-sh/uv#installation) for other methods.

### 2. Install dependencies

```bash
uv sync
```

### 3. Run the baseline model

The dataset is already included in the repository (compressed). It will automatically decompress on first use.

```bash
uv run train.py
```

This will generate training plots in the `plots/` directory showing loss and accuracy curves over epochs.

## Project Structure

- `dataset.py` - Data loading and preprocessing (auto-decompresses dataset on first use)
- `model.py` - Neural network model definition
- `train.py` - Training script
