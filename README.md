# Orwell

Orwell is a small neural network that learns the XOR truth table. It uses Python and NumPy to implement the forward pass and backpropagation for a network with two inputs, three hidden neurons, and one output. Both layers use sigmoid activations.

## Run locally

With Python 3 installed, run these commands from the repository root.

```sh
python -m pip install numpy
cd Scripts
python main.py
python test.py
```

`main.py` trains on the four XOR inputs for 10,000 epochs and writes the resulting parameters to `weights.json`. If the file already exists, training starts from its saved parameters. `test.py` loads the parameters and prints a prediction for each input. Both scripts look for `weights.json` in the current directory.

For a fresh run with randomly initialized parameters, move `weights.json` out of `Scripts` before running `main.py`.
