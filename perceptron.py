### Perceptron
import math
# X -> Dataset
# Xn -> normalized Dataset
# y -> Label
# w -> weights

# load_data(path, feature_indices, label_index)
def load_data(path, feature_indices=None, label_index=-1, skip_missing=True):
    X = []
    y = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if skip_missing and any(p == '?' for p in parts):
                continue
            if feature_indices is None:
                feats = [float(v) for i, v in enumerate(parts) if i != label_index]
            else:
                feats = [float(parts[i]) for i in feature_indices]
            # map label: 0 -> -1, non-zero -> +1 (anpassen falls nötig)
            label_value = float(parts[label_index])
            label = -1 if label_value == 0.0 else 1
            X.append(feats)
            y.append(label)
    return X, y


# Vector normalization (L2 per Feature), returns row-oriented Xn
def normalizeX(X):
    if not X:
        return []
    n_features = len(X[0])
    norms = [0.0] * n_features
    for row in X:
        for j, val in enumerate(row):
            norms[j] += val * val
    norms = [math.sqrt(n) if n > 0 else 1.0 for n in norms]
    Xn = []
    for row in X:
        Xn.append([row[j] / norms[j] for j in range(n_features)])
    return Xn
            
# dot product (necessary for the training)
def dot(u, v):
    result = 0
    for i in range(len(u)):
        result += u[i] * v[i]
    return result

# perceptron_train(Xn, y, epochs=10, lr = 1.0, add_bias = True)
def perceptron_train(Xn, y, epochs = 10, lr = 1.0, add_bias = True):
    # Kopie erstellen, um Original nicht zu ändern
    Xn = [row[:] for row in Xn]
    if add_bias:
        Xn = [row + [1.0] for row in Xn]
    n = len(Xn[0])
    w = [0.0] * n
    for epoch in range(epochs):
        updates = 0
        for xi, yi in zip(Xn, y):
            if yi * dot(w, xi) <= 0:
                for j in range(n):
                    w[j] += lr * yi * xi[j]
                updates += 1
        if updates == 0:
            break
    return w

def perceptron_predict(w, X, add_bias = True):
    # Kopie erstellen, um Original nicht zu ändern
    X = [row[:] for row in X]
    if add_bias:
        X = [row + [1.0] for row in X]
    preds = []
    for xi in X:
        preds.append(1 if dot(w, xi) > 0 else -1)
    return preds

# mathematical and statistical helping functions
def calc_average(list):
    return sum(list) / len(list) if len(list) > 0 else 0.0


# main
if __name__ == "__main__":
    print("Running perceptron.py for demo...")
    data_path = 'data/processed.cleveland.data'
    
    #Example: use three features for demo
    X, y = load_data(data_path, feature_indices=[8,11], label_index=-1)
    Xn = normalizeX(X)

    w = perceptron_train(Xn, y, epochs=50, lr=1.0)
    preds = perceptron_predict(w, Xn)
    acc = sum(1 for a,b in zip(preds, y) if a==b) / len(y) if y else 0.0
    
    print(f"Trained weights: {w}")
    print(f"Accuracy on training data: {acc:.3f}")