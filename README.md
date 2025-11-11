# Perceptron Heart Disease (minimal)

Minimal repo demonstrating a plain-Python Perceptron on the UCI Cleveland heart dataset.

Quick links
- Code: [perceptron.py](perceptron.py)
- Analysis script: [analysis.py](analysis.py)
- Simple CLI diagnoser: [diagnoser.py](diagnoser.py)
- Dataset folder: [data/](data)

How to run
1. Train / inspect: run [perceptron.py](perceptron.py) which uses [`perceptron.load_data`](perceptron.py), [`perceptron.perceptron_train`](perceptron.py) and [`perceptron.perceptron_predict`](perceptron.py).
2. Find best feature pairs: run [analysis.py](analysis.py).
3. Interactive prediction: run [diagnoser.py](diagnoser.py).

Project steps (based on CRISP‑DM)
- Business understanding: define goal (binary heart disease risk).
- Data understanding: inspect [data/processed.cleveland.data](data/).
- Data preparation: cleaning / normalization (see [`perceptron.normalizeX`](perceptron.py) usage).
- Modeling: Perceptron training (`perceptron.perceptron_train`).
- Evaluation: measure accuracy in [analysis.py](analysis.py).
- Deployment: simple CLI predictor in [diagnoser.py](diagnoser.py).

License / notes
- Educational/demo code. Validate and extend before any real-world use.