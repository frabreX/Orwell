# Orwell

A NumPy neural network that learns the XOR truth table. It uses two inputs, three hidden neurons, one output, and sigmoid activations. The training script implements the forward pass and backpropagation directly.

## Run locally

```sh
python -m pip install -r requirements.txt
python Scripts/predict_xor.py
python Scripts/train_xor.py
```

Prediction prints the model output for all four input pairs. Training runs for 10,000 epochs and writes a local checkpoint to `Scripts/weights.json`. Both scripts prefer that file and fall back to `Scripts/baseline_weights.json`. For random initialization, temporarily move both checkpoint files out of `Scripts` before training.
