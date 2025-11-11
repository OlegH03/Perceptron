from perceptron import load_data, perceptron_train, perceptron_predict

if __name__ == "__main__":
    
    data_path = "data/processed.cleveland.data"
    
    print("\nPlease provide the known numbers of health attributes separated by a \", \"")
    
    feature_names = [
        "0.age",          # 0
        "1.sex",          # 1
        "2.cp",           # 2 (chest pain type)
        "3.trestbps",     # 3 (resting blood pressure)
        "4.chol",         # 4 (cholesterin)
        "5.fbs",          # 5 (fasting blood sugar)
        "6.restecg",      # 6
        "7.thalach",      # 7 (max heart rate achieved)
        "8.exang",        # 8 (exercise induced angina)
        "9.oldpeak",      # 9 (ST depression)
        "10.slope",       # 10 (slope of peak exercise)
        "11.ca",          # 11 (number of major vessels colored by fluoroscopy)
        "12.thal",        # 12 (thalassemia)
    ]

    for feature in feature_names:
        print(feature, end = ", ")
    
    features = input("\n\n>")

    # gewünschte features in Liste holen:
    features_list = [int(f.strip()) for f in features.split(",") if f.strip()]
    
    # Validierung der Feature-Indizes
    if any(i < 0 or i >= len(feature_names) for i in features_list):
        print("Error: Feature indices must be between 0 and 12")
        exit(1)

    print("Enter concrete values of features: ")
    concrete_values = list()
    for i in features_list:
        print(feature_names[i] + ": ")
        value = input(">")
        concrete_values.append(value)

    X, y = load_data(data_path, feature_indices=features_list, label_index=-1)
    
    if not X:
        raise RuntimeError("Keine Trainingsdaten geladen.")
    if len(concrete_values) != len(X[0]):
        raise RuntimeError("Anzahl der eingegebenen Werte stimmt nicht mit der Anzahl der gewählten Features überein.")
    
    concrete_values = [float(v) for v in concrete_values]
    
    # Normen berechnen (einmalig für Training und Sample)
    n_features = len(X[0])
    norms = [0.0] * n_features
    for row in X:
        for j, val in enumerate(row):
            norms[j] += val * val
    norms = [n ** 0.5 if n > 0 else 1.0 for n in norms]
    
    # Trainingsdaten normalisieren
    Xn = [[row[j] / norms[j] for j in range(n_features)] for row in X]
    
    # Sample mit denselben Normen normalisieren
    sample_norm = [concrete_values[j] / norms[j] for j in range(n_features)]
    # sample_norm_with_bias = sample_norm + [1.0]  # Bias hinzufügen

    w = perceptron_train(Xn, y, epochs=50, lr=1.0)
    preds = perceptron_predict(w, [sample_norm])  # perceptron_predict fügt Bias selbst hinzu
    pred = preds[0]
    
    print("Prediction (raw):", pred, "(-1=healthy, 1=disease)")
    if pred == -1:
        print("No heart disease.")
    else:
        print("Probably heart disease.")

    train_preds = perceptron_predict(w, Xn)
    train_acc = sum(1 for a,b in zip(train_preds, y) if a==b) / len(y)
    print(f"Training accuracy: {train_acc:.3f}")