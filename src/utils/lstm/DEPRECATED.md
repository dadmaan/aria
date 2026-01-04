# DEPRECATED: LSTM Utils Module

**Status:** DEPRECATED as of 2025-11-11

## What Happened?

The `lstm_utils.py` module has been deprecated and archived as part of the PyTorch/SB3 isolation cleanup.

## Why?

This module was entirely dependent on TensorFlow/Keras libraries:
- `tensorflow.keras` for model operations
- `kerastuner` for hyperparameter tuning
- TensorFlow-specific wrappers and utilities

As part of our migration to a PyTorch/Stable-Baselines3 ecosystem, we've removed all TensorFlow dependencies from the codebase.

## Where Did It Go?

The original code has been moved to:
```
/archive/deprecated_tf_modules/lstm_utils.py
```

## What Should You Do?

If you need LSTM functionality:

1. **For PyTorch LSTM operations**, use:
   - `torch.nn.LSTM` for basic LSTM layers
   - `torch.nn.utils.rnn` for sequence packing/padding
   - Native PyTorch embedding layers

2. **For hyperparameter tuning**, consider:
   - Optuna (PyTorch-friendly)
   - Ray Tune
   - Weights & Biases sweeps

3. **For visualization**, use:
   - TensorBoard with PyTorch
   - Matplotlib/Plotly for custom visualizations

## Migration Resources

- PyTorch LSTM Tutorial: https://pytorch.org/docs/stable/nn.html#lstm
- Optuna with PyTorch: https://optuna.readthedocs.io/
- PyTorch Embedding: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

---
*For questions or migration assistance, refer to the project documentation or the archived code.*
