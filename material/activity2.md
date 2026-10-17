# Activity 2: Building an MLP with Keras: Solving XOR

In Activity 1, we saw that a single perceptron can represent AND and OR, but cannot solve XOR. Can a network with a **hidden layer** do better?

In this activity, you will build a small Multi-Layer Perceptron (MLP) using Keras, train it on the four XOR examples, and inspect its predictions. We will use training commands today, but study **how training works internally** in a later session.

**Learning objectives**

- Explain why an MLP with nonlinear activations can represent XOR.
- Identify the input, hidden, and output layers in a Keras model.
- Build an MLP using `Sequential`, `Input`, and `Dense`.
- Use `compile()`, `fit()`, and `predict()` at a high level.
- Compare predictions with the XOR truth table and experiment with the architecture.

<!-- ![](./img/keras.png) -->
<img src="./img/keras.png" width="50%">

> Keras provides a high-level API for building neural networks. TensorFlow, PyTorch and JAX provide the underlying computational frameworks. CPU, GPU and TPU are hardware on which the computations can run.


## 1. Prepare the XOR data

**Before running the code:** How many input features does each example have? How many examples are there?

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# XOR truth table: inputs (X) and expected outputs (y)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)
print(X)
print(y)
```

**Your task:** Explain what `(4, 2)` and `(4, 1)` mean.

<details>
<summary><b>Explanation</b></summary>

- `X` has 4 examples and 2 features per example.
- `y` has 4 expected outputs, one for each example.

</details>

## 2. Build the MLP

A single perceptron draws one straight decision boundary. An MLP combines computations from several neurons. **Nonlinear activation functions** are what allow the network to represent more complex boundaries.

Our model has **2 inputs → 4 hidden neurons → 1 output**.

**Before running the code:** Which line creates the hidden layer? Why does the final layer have just one neuron?

```python
model = keras.Sequential(name="MLP_XOR_Solver")
model.add(keras.Input(shape=(2,)))
model.add(layers.Dense(units=4, activation="relu"))
model.add(layers.Dense(units=1, activation="sigmoid"))

model.summary()
```

<details>
<summary><b>Code explanation</b></summary>

- `Sequential`: stacks layers in order.
- `Input(shape=(2,))`: each example contains two input values.
- `Dense(units=4, activation="relu")`: a fully connected hidden layer with four neurons; ReLU introduces nonlinearity.
- `Dense(units=1, activation="sigmoid")`: produces one value between 0 and 1, suitable for a binary prediction.
- `summary()`: shows the architecture and number of trainable parameters.

</details>

**Your task:** Use `model.summary()` to calculate the number of weights and biases. What is the total number of trainable parameters?

<details>
<summary><b>Sample solution: parameter counts</b></summary>

**Hidden layer**

- Weights: \(2 \times 4 = 8\)
- Biases: \(4\)
- Total: **12**

**Output layer**

- Weights: \(4 \times 1 = 4\)
- Biases: \(1\)
- Total: **5**

**Network total: 17 trainable parameters.**

</details>

## 3. Prepare the model for training

We tell Keras how to measure mistakes, how to update the model, and which performance measure to report. **We will explain the mathematics behind these choices in a future session.**

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

<details>
<summary><b>At a glance: what does compile() specify?</b></summary>

- **Optimizer (`adam`)**: the method Keras uses to adjust the model's parameters.
- **Loss (`binary_crossentropy`)**: measures how far predictions are from the correct binary labels.
- **Metric (`accuracy`)**: reports how often the model predicts the correct class.

For now, focus on the role of each setting, not its internal mathematics.

</details>

**Check your understanding:** Which setting determines the score printed as *accuracy*?

## 4. Train the model

Initially, the model's weights are not tuned for XOR. `fit()` uses the examples and labels to adjust those weights.

```python
# Try changing this number and rebuilding the model to compare results.
num_epochs = 1000

history = model.fit(
    X, y,
    epochs=num_epochs,
    batch_size=4,
    verbose=0
)

print(f"Final training accuracy: {history.history['accuracy'][-1]:.0%}")
```

<details>
<summary><b>At a glance: what does fit() do?</b></summary>

- `fit(X, y, ...)`: trains the model using inputs and expected outputs.
- An **epoch** is one pass through all four examples.
- `batch_size=4`: all four XOR examples are handled together per training step.
- `verbose=0`: hides the progress display.

The internal learning process, including backpropagation and optimization, comes later in the course.

</details>

**Your task:** Record the reported accuracy. Is it always 100% when you rebuild and retrain the model?

> [!NOTE]
> This tiny example does **not** guarantee a perfect result. Different random initial weights can lead to different outcomes, and some runs may fail to learn XOR. That is an interesting observation, not an indication that your code is necessarily incorrect.

## 5. Inspect the predictions

Accuracy gives a summary. Looking at each input and its predicted output tells us more.

```python
probabilities = model.predict(X, verbose=0).flatten()
predictions = (probabilities >= 0.5).astype(int)

for inputs, probability, predicted, expected in zip(
    X, probabilities, predictions, y.flatten()
):
    print(
        f"Input: {inputs.astype(int)} | "
        f"Score: {probability:.3f} | "
        f"Predicted: {predicted} | Expected: {int(expected)}"
    )
```

**Your tasks:**

1. Which XOR inputs were classified correctly?
2. Why do we convert the sigmoid score to 0 or 1 using a threshold of 0.5?
3. How is this MLP different from the single perceptron in Activity 1?

<details>
<summary><b>Discussion prompts / sample answers</b></summary>

- The output layer produces a score between 0 and 1; the 0.5 threshold converts it into one of two classes.
- A single perceptron can only create a linear decision boundary.
- An MLP with a hidden layer and nonlinear activations can represent XOR, although a particular training run may not find a successful set of weights.

</details>

## 6. Experiment: change the architecture

**Choose one experiment.** Before rerunning the model, predict what might change.

- Change the hidden layer from `units=4` to `units=2` or `units=8`.
- Replace `relu` with `tanh` in the hidden layer.
- Change `num_epochs` from `1000` to `2000`.

**Important:** Rebuild the model from Section 2 and rerun Sections 3–5 for each experiment. Otherwise you would continue training the existing model, which makes the comparison harder to interpret.

| Experiment | Prediction before running | Final accuracy | What did you observe? |
|---|---|---|---|
| Original model | | | |
| Changed model | | | |

**Discussion:** Did the change improve results? Can one run establish that an architecture is always better?

<details>
<summary><b>Instructor note: interpreting variable results</b></summary>

A single result is not enough to compare architectures reliably because model initialization is random. Students need only notice that architecture and training choices affect the result. We will study these effects more rigorously later.

</details>

## Conclusion

A single perceptron cannot solve XOR because XOR is not linearly separable. Adding hidden neurons and **nonlinear activation functions** gives an MLP the ability to represent the XOR relationship.

Today we used Keras to create, train, and test that MLP. **Next week, we will look more closely at how its weights are learned.**

**Exit question:** What is the most important difference between a single perceptron and the MLP you built?
