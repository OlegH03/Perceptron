# Perceptron Heart Disease (minimal)

Minimal repo demonstrating a plain-Python Perceptron (Single neuron) on the UCI Cleveland heart dataset.

## Quick links
- Code: [perceptron.py](perceptron.py)
- Analysis script: [analysis.py](analysis.py)
- Simple CLI diagnoser: [diagnoser.py](diagnoser.py)
- Data: https://archive.ics.uci.edu/dataset/45/heart+disease

## How to run
1. Train / inspect: run [perceptron.py](perceptron.py) which uses [`perceptron.load_data`](perceptron.py), [`perceptron.perceptron_train`](perceptron.py) and [`perceptron.perceptron_predict`](perceptron.py).
2. Find best feature pairs: run [analysis.py](analysis.py).
3. Interactive prediction: run [diagnoser.py](diagnoser.py).

## Project steps
- Business understanding: define goal (binary heart disease risk).
- Data understanding: inspect [data/processed.cleveland.data](data/).
- Data preparation: cleaning / normalization (see [`perceptron.normalizeX`](perceptron.py) usage).
- Modeling: Perceptron training (`perceptron.perceptron_train`).
- Evaluation: measure accuracy in [analysis.py](analysis.py).
- Deployment: simple CLI predictor in [diagnoser.py](diagnoser.py).

## How to get the training data
To run the project, please download the `processed.cleveland.data` file (or your specific file) from the original UCI link and place it into a new folder named `./data/` in the root directory of this repository.

1.  **Download:** Go to the [Heart Disease Data Set](https://archive.ics.uci.edu/dataset/45/heart+disease) page.
2.  **Save as:** Save the data file into the `./data/` folder.

License / notes
- Educational/demo code. Validate and extend before any real-world use.