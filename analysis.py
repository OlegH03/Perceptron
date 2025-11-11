from perceptron import load_data, normalizeX, perceptron_train, perceptron_predict
import itertools

feature_names = [
    "age",          # 0
    "sex",          # 1
    "cp",           # 2 (chest pain type)
    "trestbps",     # 3 (resting blood pressure)
    "chol",         # 4 (cholesterin)
    "fbs",          # 5 (fasting blood sugar)
    "restecg",      # 6
    "thalach",       # 7 (max heart rate achieved)
    "exang",        # 8 (exercise induced angina)
    "oldpeak",      # 9 (ST depression)
    "slope",        # 10 (slope of peak exercise)
    "ca",           # 11 (number of major vessels colored by fluoroscopy)
    "thal",         # 12 (thalassemia)
]

data_path = "data/processed.cleveland.data"

X_full, y = load_data(data_path, feature_indices=None, label_index=-1)

print("Number of features loaded:", len(X_full[0]))
print("Number of feature names: ", len(feature_names))

results = []

for i, j in itertools.combinations(range(len(X_full[0]) - 1), 2):
    # build X_subset just with feature i and j
    X_subset = [[row[i], row[j]] for row in X_full]

    # normalize per feature:
    Xn = normalizeX(X_subset)

    # train perceptron
    w = perceptron_train(Xn, y, epochs = 50, lr = 1.0)

    # eval
    preds = perceptron_predict(w, Xn)
    acc = sum(1 for a,b in zip(preds, y) if a==b) / len(y)

    f1_name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
    f2_name = feature_names[j] if j < len(feature_names) else f"feat_{j}"

    results.append({
        "f1_idx": i,
        "f2_idx": j,
        "f1_name": f1_name,
        "f2_name": f2_name,
        "acc": acc,
        "w": w
    })


# sort for accuracy
results.sort(key = lambda r: r["acc"], reverse=True)

best = results[0]

print("Best feature pair:")
print(best["f1_name"], "+", best["f2_name"], " -> Acc =", round(best["acc"], 3))
print("Weights (incl. bias):", best["w"])