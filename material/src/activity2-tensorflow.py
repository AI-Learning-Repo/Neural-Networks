import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices("GPU"))

# XOR data
X = np.array([
    [0, 0], [0, 1], [1, 0], [1, 1]
], dtype=np.float32)

y = np.array([
    [0], [1], [1], [0]
], dtype=np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Set random seed for reproducibility
seed = 32
tf.keras.utils.set_random_seed(seed)

# Define Model using TensorFlow Sequential API
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(4, activation="relu", name="hidden_layer"),
    tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer")
], name="MLP_XOR_Solver")

model.summary()  # 12 + 5 = 17 trainable parameters

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5)]
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