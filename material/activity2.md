# Activity 2: Building an MLP with Keras — Solving XOR

In **Activity 1**, you manually selected the weights and bias of a single perceptron. You found that a perceptron can represent AND and OR but **cannot represent XOR**, because XOR is not linearly separable.

In this activity, you will build a **multi-layer perceptron (MLP)** with a hidden layer. Instead of choosing its weights yourself, you will use **Keras to learn them from examples**. We will explore what training does, but leave the mathematics of loss minimization, gradients, and backpropagation for a later lesson.

## Learning objectives

By the end of this activity, you should be able to:

1. Explain why hidden neurons with nonlinear activation functions make XOR representable.
2. Build a Keras MLP using `Sequential`, `Input`, and `Dense`.
3. Count weights and biases in a simple MLP.
4. Describe, at a high level, the roles of `compile()`, `fit()`, `evaluate()`, and `predict()`.
5. Interpret output scores, predicted classes, and training accuracy.
6. Change one model setting, compare results, and explain why a single run proves little.

**How to work:** Make a prediction before running each section. Complete the questions first, then open the **Solution / explanation** sections to check your understanding. Run the code in order in a Python notebook or script.

**Software:** NumPy and **Keras 3** with the **TensorFlow backend**, consistent with the Keras approach used in Activities 3 and 4. If needed, install them in your Python environment:

```bash
python -m pip install numpy keras tensorflow
```

Keras is a high-level API for defining and training networks. TensorFlow is the backend we will use to perform the computations. Keras 3 can also use other supported backends, but switching backends is **not required** for this lab.

<!-- ![](./img/keras.png) -->
<img src="./img/keras.png" width="50%">

> **Important:** Our dataset has only **four examples**, and we will train and inspect the model on those *same four examples*. The resulting accuracy is **training accuracy**, not evidence that the network generalizes to unseen data. We will introduce train/validation/test splits in a later activity.

---

## 1. Prepare the XOR data

**Before running:** How many examples does the XOR truth table contain? How many input features does each example have?

```python
import numpy as np
import keras
from keras import layers

print("Keras version:", keras.__version__)
print("Keras backend:", keras.config.backend())

# Four rows of the XOR truth table.
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Inputs:\n", X)
print("Labels:\n", y)
```

**Your tasks**

1. Explain what `X.shape == (4, 2)` means.
2. Explain what `y.shape == (4, 1)` means.
3. Why is XOR a **binary classification** problem?

<details>
<summary><strong>Solution / explanation: XOR data</strong></summary>

1. `X` contains **4 examples**, each with **2 input features** (`x1`, `x2`).
2. `y` contains **4 labels**, one binary label per example.
3. We want to predict one of two classes: `0` (inputs are equal) or `1` (inputs differ). We are not predicting an unrestricted numerical quantity.

The correct input/output pairs are `(0,0) → 0`, `(0,1) → 1`, `(1,0) → 1`, and `(1,1) → 0`.

</details>

---

## 2. Build the MLP

A single perceptron can draw only one straight decision boundary. An MLP combines several neurons, and **nonlinear hidden-layer activations** allow it to represent relationships that one straight boundary cannot capture.

We will use the architecture:

```text
2 input features  →  4 hidden neurons (ReLU)  →  1 output neuron (sigmoid)
```

The hidden layer uses **ReLU**, which returns `0` for a negative input and otherwise returns the input. The output neuron uses **sigmoid**, which produces a score between `0` and `1`.

**Before running:** Which line creates the hidden layer? Why is there only one output neuron?

```python
# Use a fixed seed so repeated runs in a similar environment
# are easier to compare. A seed does NOT guarantee 100% accuracy.
seed = 32
keras.utils.set_random_seed(seed)

model = keras.Sequential(name="MLP_XOR_Solver")
model.add(keras.Input(shape=(2,)))
model.add(layers.Dense(units=4, activation="relu", name="hidden_layer"))
model.add(layers.Dense(units=1, activation="sigmoid", name="output_layer"))

model.summary()
```

**Your tasks**

1. Identify the input, hidden, and output parts of the model.
2. Use `model.summary()` to calculate **weights**, **biases**, and **total trainable parameters** for each `Dense` layer.
3. Why is the hidden-layer activation important for XOR?

**Hint:** A `Dense` layer with `n` incoming values and `m` neurons has `n × m` weights and `m` biases.

<details>
<summary><strong>Solution / explanation: architecture and parameters</strong></summary>

- `keras.Input(shape=(2,))` describes two input features; it does **not** add trainable parameters.
- `Dense(units=4, activation="relu")` is the hidden layer.
- `Dense(units=1, activation="sigmoid")` is the output layer. One output score is sufficient for a binary classification task.
- A nonlinear activation is essential here: without nonlinear hidden activations, a stack of `Dense` layers would still be equivalent to a single linear transformation, which cannot represent XOR.

| Layer | Weights | Biases | Parameters |
|---|---:|---:|---:|
| Hidden: 2 inputs → 4 neurons | `2 × 4 = 8` | `4` | **12** |
| Output: 4 inputs → 1 neuron | `4 × 1 = 4` | `1` | **5** |
| **Total** | **12** | **5** | **17** |

In symbols, the network first calculates hidden activations and then an output score:

<!-- $$
h = \mathrm{ReLU}(xW_1 + b_1), \qquad \hat{y} = \sigma(hW_2 + b_2)
$$ -->


```math
h = \operatorname{ReLU}(xW_1 + b_1), \qquad \hat{y} = \sigma(hW_2 + b_2)
```

Here, $\sigma$ is sigmoid. You **do not** need to calculate the matrix multiplications by hand in this activity.

</details>

---

## 3. Compile the model

Before training, Keras needs to know:

- **Optimizer:** the method used to adjust weights (here, Adam).
- **Loss:** the quantity the optimizer tries to reduce (here, binary cross-entropy).
- **Metric:** the score we report to understand performance (here, binary accuracy).

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5)]
)
```

**Your tasks**

1. Which setting tells Keras what to minimize during training?
2. Which setting tells Keras to report the fraction of correctly classified examples?
3. Why is **binary** accuracy suitable here?

<details>
<summary><strong>Solution / explanation: compile()</strong></summary>

1. `loss="binary_crossentropy"` specifies the loss that training aims to reduce.
2. `keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5)` specifies the reported accuracy. Using an explicit metric also makes the classification threshold clear.
3. Each target is one of two classes, `0` and `1`. The metric converts sigmoid scores into binary predictions and checks them against the labels.

The **optimizer** updates parameters using information from the loss. The **accuracy metric** helps us interpret results; it is **not** the quantity Adam directly minimizes here.

We will study the mathematics of optimization later.

</details>

---

## 4. Train the MLP

Initially, the model's weights are not chosen for XOR. During training, `fit()` repeatedly uses the examples and their labels to adjust the weights.

```python
num_epochs = 1000

history = model.fit(
    X,
    y,
    epochs=num_epochs,
    batch_size=4,
    shuffle=False,
    verbose=0
)

# Evaluate the final trained model on the four training examples.
train_loss, train_accuracy = model.evaluate(X, y, verbose=0)

print(f"Final training loss: {train_loss:.4f}")
print(f"Final training accuracy: {train_accuracy:.0%}")
```

**Your tasks**

1. What is an **epoch**? What does `batch_size=4` mean for this dataset?
2. Record your training accuracy. If it is 75%, how many examples were classified correctly?
3. Is 100% accuracy guaranteed every time? Does a lower score prove that MLPs cannot solve XOR?
4. Why should we **not** call this result *test accuracy*?

> **Note:** A model with this architecture **can** represent XOR, but a particular training run may not learn suitable weights. Do not change the recorded outcome just to make it equal 100%.

<details>
<summary><strong>Solution / explanation: training</strong></summary>

1. One **epoch** is one pass through the four training examples. Because `batch_size=4`, the entire XOR dataset is used together in each training update (**full-batch training**).
2. **75%** means **3 of the 4** XOR examples were correctly classified. **50%** means 2 out of 4.
3. **No.** Weights start from an initialization, and training may fail to reach a correct solution within the selected number of epochs. A disappointing result does **not necessarily** mean the code is wrong, and it does not mean the architecture is incapable of representing XOR.
4. We used the **same four examples for fitting and evaluation**. Nothing has been reserved as unseen test data.

**Key distinction:**

- **Representation:** Does there exist a set of weights that makes the network solve XOR? **Yes.**
- **Training/optimization:** Does this particular training run find such weights? **Not always.**

The seed helps make classroom results more repeatable in similar environments, but hardware and software differences can still affect training.

</details>

---

## 5. Inspect each prediction

A single accuracy number does not show which input combinations were classified correctly. Let's inspect all four scores and predicted classes.

```python
scores = model.predict(X, verbose=0).reshape(-1)

# Use the same decision rule as BinaryAccuracy with threshold=0.5.
predictions = (scores > 0.5).astype(int)

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

**Your tasks**

1. Which XOR inputs were classified correctly? Which, if any, were wrong?
2. What is the difference between a **sigmoid score** and a **predicted class**?
3. Why do we use a threshold of `0.5`?
4. Does the manually calculated accuracy match `model.evaluate()`?
5. How is this MLP different from the single perceptron in Activity 1?

<details>
<summary><strong>Solution / explanation: predictions</strong></summary>

1. **It depends on your run.** Compare each printed prediction with its expected value. If training succeeded perfectly, all four pairs match: `0, 1, 1, 0`.
2. Sigmoid produces a continuous score between 0 and 1. We then apply a threshold to choose class `0` or `1`. Such a score is often interpreted as a model-estimated probability, but it is not automatically guaranteed to be well calibrated.
3. A score **greater than 0.5** becomes class `1`; a score at or below 0.5 becomes class `0`. This is the convention used by the explicit `BinaryAccuracy` metric in Section 3.
4. **Yes**, allowing for rounding of displayed percentages. Both use the same four examples and the same threshold.
5. One step-function perceptron has one linear decision boundary and cannot represent XOR. An MLP has a hidden layer with nonlinear activations and **can** represent XOR, although one training run may still fail.

A possible **successful** run might show low scores for `[0,0]` and `[1,1]`, and high scores for `[0,1]` and `[1,0]`. **The exact decimal scores are not fixed expected output.**

</details>

---

## 6. Experiment: change one setting

**Choose one change**, then predict its possible effect before running it:

- **Capacity:** change `units=4` to `units=2` or `units=8`.
- **Activation:** replace hidden-layer `relu` with `tanh`.
- **Training time:** increase `num_epochs` from `1000` to `2000`.

**Important for a fair comparison:** Go back to Section 2 and **build a fresh model**, then re-run Sections 3–5. Do not just call `fit()` again on the already-trained model: that would continue training from its previous weights. For the first comparison, keep `seed = 32` unchanged and modify only **one** setting.

| Configuration | Prediction before running | Final training accuracy | Final training loss | What happened? |
|---|---|---|---|---|
| Original: 4 ReLU units, 1000 epochs | | | | |
| Your modified configuration | | | | |

**Your tasks**

1. Did the modification help on **your** run? Use the actual accuracy and loss values.
2. If both models have 100% accuracy, can their losses still differ?
3. Can a single run establish that one architecture is **always** better?
4. *(Optional)* Try multiple seeds, rebuilding a fresh model for each seed. Are the outcomes identical?

<details>
<summary><strong>Solution / explanation: architecture experiments</strong></summary>

There is **no single correct results table**: model outcomes can vary with initialization and environment. Report actual measurements, not a prescribed 100% score.

Possible interpretations:

- **2 hidden units:** the network still has enough representational capacity for XOR, but training can be sensitive to the starting weights.
- **8 hidden units:** the model has more trainable parameters and might learn XOR more reliably in some trials; **it is not guaranteed** to be faster or better in every run.
- **`tanh` instead of `relu`:** both are nonlinear and can represent XOR. Results may differ because the activations behave differently; neither is universally superior.
- **2000 instead of 1000 epochs:** extra training steps sometimes reduce loss or change predictions, but they do not guarantee success and cannot necessarily fix ineffective optimization.

Parameter counts if you choose the capacity experiment:

| Architecture | Trainable parameters |
|---|---:|
| 2 inputs → **2** hidden → 1 output | `(2 × 2 + 2) + (2 × 1 + 1) = 9` |
| 2 inputs → **4** hidden → 1 output | **17** |
| 2 inputs → **8** hidden → 1 output | `(2 × 8 + 8) + (8 × 1 + 1) = 33` |

**Answers to the discussion questions:**

1. Judge improvement from the measurements you actually obtained. A change in one run is an **observation**, not a universal rule.
2. **Yes.** Accuracy measures whether each class is on the correct side of the threshold. Binary cross-entropy also responds to *how strongly* scores support the labels.
3. **No.** Repeating experiments with multiple initial seeds is more informative than relying on one result.
4. Results may change between seeds. Identical results are possible too; they are not guaranteed.

**Optional code to run fresh, repeatable trials** (run *after* the original Sections 1–5):

```python
def run_trial(hidden_units=4, activation="relu", epochs=1000, seed=32):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)

    candidate = keras.Sequential([
        keras.Input(shape=(2,)),
        layers.Dense(hidden_units, activation=activation),
        layers.Dense(1, activation="sigmoid")
    ])
    candidate.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5)]
    )
    candidate.fit(X, y, epochs=epochs, batch_size=4,
                  shuffle=False, verbose=0)
    loss, accuracy = candidate.evaluate(X, y, verbose=0)
    return accuracy, loss

for trial_seed in [7, 21, 32]:
    accuracy, loss = run_trial(hidden_units=4, seed=trial_seed)
    print(f"Seed {trial_seed}: accuracy={accuracy:.0%}, loss={loss:.4f}")
```

Rebuilding and reseeding makes each trial a fresh training run. Keep other settings fixed when comparing a single change.

</details>

---

## 7. Troubleshooting and concept check

**Question 1.** Your model achieves 75% accuracy. Is its code necessarily wrong?

**Question 2.** Why does the fact that an MLP *can* represent XOR not guarantee that `fit()` will reach 100%?

**Question 3.** What would you try if your model stays at 50% or 75%?

<details>
<summary><strong>Solution / explanation: troubleshooting</strong></summary>

**1. Not necessarily.** A score of 75% means 3 out of 4 labels were correctly predicted. First check the data, model structure, compilation, and classification threshold. If these are correct, the run may simply have learned an imperfect solution.

**2.** A suitable combination of weights **exists**, but the training process begins from particular initial weights and has a limited number of updates. Being able to represent a function is different from successfully learning its weights.

**3. Try one change at a time:**

- Rebuild the model and try a different random seed.
- Try `tanh` in the hidden layer.
- Try more hidden units.
- Try more epochs, especially if loss is still decreasing.

**Do not automatically conclude** that any one of these changes will solve the problem. Compare the actual predictions and loss.

**Optional background (not required for this lesson):** With ReLU, a hidden unit whose pre-activation is nonpositive on every training example may receive no useful gradient signal from those examples. This is one possible reason that some initializations learn poorly. It does **not** follow that every run below 100% is caused by a “dead ReLU,” nor that adding epochs can always resolve the problem. The details belong in a later lesson about optimization and backpropagation.

</details>

---

## 8. Wrap-up

A single perceptron cannot represent XOR with one linear decision boundary. An MLP with hidden neurons and nonlinear activations **can** represent XOR. In this activity, you defined such a network with Keras, trained it, inspected its four predictions, and experimented with one setting.

**Exit question:** What is the single most important architectural difference between the perceptron in Activity 1 and the MLP in this activity?

<details>
<summary><strong>Sample answer: exit question</strong></summary>

The MLP contains a **hidden layer with nonlinear activation functions**, allowing it to represent nonlinear relationships such as XOR. A single perceptron cannot.

</details>

---

## Complete runnable solution (Sections 1–5)

The following script repeats the baseline workflow from Sections 1–5, in one place. Use it to check your work **after** attempting the activity. The experiment in Section 6 is open-ended; a reproducible experiment helper is included in that section's solution above.



<details>
<summary><strong>Show complete Python solution</strong></summary>

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

</details>

### Optional references

- [Keras 3: NumPy/Python utilities (`set_random_seed`)](https://keras.io/api/utils/python_utils/)
- [Keras 3: Accuracy metrics (`BinaryAccuracy`)](https://keras.io/api/metrics/accuracy_metrics/)

**Next activity:** Move from four synthetic examples to a real dataset and learn how to prepare features, train an MLP for regression, and evaluate predictions on data not used for training.
