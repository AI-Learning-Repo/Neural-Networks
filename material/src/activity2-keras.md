# Complete solution

```python
import numpy as np
import keras
from keras import layers

print("Keras version:", keras.__version__)
print("Keras backend:", keras.config.backend())

# XOR data
X = np.array([
    [0, 0], [0, 1], [1, 0], [1, 1]
], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)
print("X shape:", X.shape)
print("y shape:", y.shape)

# Fresh model
seed = 32
keras.utils.set_random_seed(seed)
model = keras.Sequential(name="MLP_XOR_Solver")
model.add(keras.Input(shape=(2,)))
model.add(layers.Dense(4, activation="relu", name="hidden_layer"))
model.add(layers.Dense(1, activation="sigmoid", name="output_layer"))
model.summary()  # 12 + 5 = 17 trainable parameters

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5)]
)

# Train
history = model.fit(
    X, y,
    epochs=1000,
    batch_size=4,
    shuffle=False,
    verbose=0
)

# Evaluate training performance
train_loss, train_accuracy = model.evaluate(X, y, verbose=0)
print(f"Final training loss: {train_loss:.4f}")
print(f"Final training accuracy: {train_accuracy:.0%}")

# Inspect every example
scores = model.predict(X, verbose=0).reshape(-1)
predictions = (scores > 0.5).astype(int)
print("\nXOR predictions:")
for inputs, score, predicted, expected in zip(
    X, scores, predictions, y.reshape(-1)
):
    print(
        f"Input: {inputs.astype(int)} | "
        f"Score: {score:.3f} | "
        f"Predicted: {predicted} | Expected: {int(expected)}"
    )

calculated_accuracy = np.mean(predictions == y.reshape(-1))
print(f"Accuracy calculated from predictions: {calculated_accuracy:.0%}")
```

**Expected stable facts:** `X.shape` is `(4, 2)`, `y.shape` is `(4, 1)`, and the model has **17 trainable parameters**. **Accuracy, loss, and individual sigmoid scores are measured outcomes, not guaranteed fixed values.**