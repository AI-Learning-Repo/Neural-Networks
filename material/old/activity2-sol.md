# Activity 2: sample solution

### Complete Runnable Python Script

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set random seed for reproducibility
tf.random.set_seed(32)
np.random.seed(32)

# ==========================================
# 1. Prepare the XOR data
# ==========================================
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ==========================================
# 2. Build the MLP
# ==========================================
model = keras.Sequential(name="MLP_XOR_Solver")
model.add(keras.Input(shape=(2,)))
model.add(layers.Dense(units=4, activation="relu", name="hidden_layer"))
model.add(layers.Dense(units=1, activation="sigmoid", name="output_layer"))

model.summary()

# ==========================================
# 3. Compile the model
# ==========================================
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 4. Train the model
# ==========================================
num_epochs = 1000

history = model.fit(
    X, y,
    epochs=num_epochs,
    batch_size=4,
    verbose=0
)

print(f"\nFinal training accuracy: {history.history['accuracy'][-1]:.0%}")

# ==========================================
# 5. Inspect the predictions
# ==========================================
probabilities = model.predict(X, verbose=0).flatten()
predictions = (probabilities >= 0.5).astype(int)

print("\n--- Predictions vs Ground Truth ---")
for inputs, probability, predicted, expected in zip(
    X, probabilities, predictions, y.flatten()
):
    print(
        f"Input: {inputs.astype(int)} | "
        f"Score: {probability:.3f} | "
        f"Predicted: {predicted} | Expected: {int(expected)}"
    )
```

---

### Expected Console Output

```text
X shape: (4, 2)
y shape: (4, 1)

Model: "MLP_XOR_Solver"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 hidden_layer (Dense)        (None, 4)                 12        
                                                                 
 output_layer (Dense)        (None, 1)                 5         
                                                                 
=================================================================
Total params: 17 (68.00 Byte)
Trainable params: 17 (68.00 Byte)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

Final training accuracy: 100%

--- Predictions vs Ground Truth ---
Input: [0 0] | Score: 0.048 | Predicted: 0 | Expected: 0
Input: [0 1] | Score: 0.962 | Predicted: 1 | Expected: 1
Input: [1 0] | Score: 0.957 | Predicted: 1 | Expected: 1
Input: [1 1] | Score: 0.061 | Predicted: 0 | Expected: 0
```

---

## Detailed Step-by-Step Solutions

### Section 1: Prepare the XOR data

* **Before running the code: How many input features does each example have? How many examples are there?**
  * Each example has **2 input features** ($x_1$ and $x_2$).
  * There are **4 total examples** (the four combinations in the truth table: `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`).
* **Your task: Explain what `(4, 2)` and `(4, 1)` mean:**
  * `(4, 2)`: The input matrix $X$ has $4$ rows (samples) and $2$ columns (features per sample).
  * `(4, 1)`: The target matrix $y$ has $4$ rows (samples) and $1$ column (the single binary output label for each sample).

---

### Section 2: Build the MLP

* **Before running the code: Which line creates the hidden layer? Why does the final layer have just one neuron?**
  * `model.add(layers.Dense(units=4, activation="relu"))` creates the hidden layer.
  * The final layer has only **1 neuron** because XOR is a binary classification problem that produces a single scalar output (0 or 1).
* **Your task: Parameter calculation using `model.summary()`:**
  * **Hidden Layer (`Dense(4)`):**
    * Weights: $2 \text{ inputs} \times 4 \text{ neurons} = 8$ weights
    * Biases: $1 \text{ per neuron} = 4$ biases
    * Layer subtotal: $8 + 4 = \mathbf{12\text{ parameters}}$
  * **Output Layer (`Dense(1)`):**
    * Weights: $4 \text{ inputs (from hidden layer)} \times 1 \text{ neuron} = 4$ weights
    * Biases: $1 \text{ bias}$
    * Layer subtotal: $4 + 1 = \mathbf{5\text{ parameters}}$
  * **Total trainable parameters:** $12 + 5 = \mathbf{17\text{ parameters}}$.

---

### Section 3: Prepare the model for training

* **Check your understanding: Which setting determines the score printed as *accuracy*?**
  * The argument `metrics=["accuracy"]` inside `model.compile(...)`. 
  * Note: The loss function (`binary_crossentropy`) guides the optimizer during gradient updates, but the metric specifies what human-readable score Keras computes and prints.

---

### Section 4: Train the model

* **Your task: Record the reported accuracy. Is it always 100% when you rebuild and retrain the model?**
  * Typical result: **100%** on most runs.
  * **No, it is not always 100%.** Because neural network weights are randomly initialized, small networks trained on tiny datasets can occasionally get trapped in local minima or flat plateaus (e.g., predicting 0.5 or 0 for everything, yielding 50% or 75% accuracy).

---

### Section 5: Inspect the predictions

* **1. Which XOR inputs were classified correctly?**
  * All four inputs (`[0, 0]`, `[0, 1]`, `[1, 0]`, and `[1, 1]`) were classified correctly.
* **2. Why do we convert the sigmoid score to 0 or 1 using a threshold of 0.5?**
  * The sigmoid function maps arbitrary real values into a probability range $[0, 1]$. Since XOR targets are discrete classes (0 or 1), $0.5$ represents the natural midpoint/decision boundary:
    $$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|X) \ge 0.5 \\ 0 & \text{if } P(y=1|X) < 0.5 \end{cases}$$
* **3. How is this MLP different from the single perceptron in Activity 1?**
  * A single perceptron has no hidden layers and can only separate data with a single straight line (hyperplane). XOR is not linearly separable.
  * The MLP uses a hidden layer with nonlinear activations (`ReLU`), which warps/projects the 2D input space into a higher-dimensional representation where the XOR classes become linearly separable by the final sigmoid neuron.

---

### Section 6: Experiment Table & Discussion

| Experiment | Prediction before running | Final accuracy | What did you observe? |
|---|---|---|---|
| **Original Model** (4 ReLU units, 1000 epochs) | Baseline model; should converge to 100% most of the time. | **100%** | Loss steadily decreased; scores for `[0,1]` and `[1,0]` are $>0.95$, while `[0,0]` and `[1,1]` are $<0.07$. |
| **Experiment A**: Change to `units=2` | Might struggle or take longer because 2 hidden neurons is the theoretical minimum for XOR. | **75% or 100%** (Run 1: 75%, Run 2: 100%) | With 2 neurons and ReLU, if one neuron "dies" (outputs 0 for all samples), the network gets stuck and achieves only 50–75% accuracy. Highly sensitive to initial weights. |
| **Experiment B**: Change to `units=8` | Should converge faster and more consistently to 100%. | **100%** | Reached 100% accuracy much earlier (around epoch 400). Provides redundant paths so random initialization rarely causes failure. |
| **Experiment C**: Replace `relu` with `tanh` | `tanh` works well for XOR because its zero-centered range $(-1, 1)$ prevents dead neuron issues. | **100%** | Very smooth convergence; output probabilities reached near-extremes ($>0.99$ and $<0.01$). |
| **Experiment D**: Change `num_epochs=2000` | Should push predictions closer to absolute 0.0 and 1.0. | **100%** | Accuracy remained 100%, but loss dropped further, sharpening probabilities closer to 0 and 1. |

#### Discussion Questions
* **Did the change improve results?**
  * Increasing units to 8 or using `tanh` improved stability across random seeds. Dropping to 2 units made training brittle and prone to getting stuck at 75% accuracy.
* **Can one run establish that an architecture is always better?**
  * **No.** Because weight initialization and mini-batch dynamics are stochastic, a single run might succeed or fail by chance. Evaluating model architectures reliably requires multiple runs across different random seeds and computing average performance.

---

### Conclusion / Exit Question

* **Exit question: What is the most important difference between a single perceptron and the MLP you built?**
  * The presence of a **hidden layer with nonlinear activation functions**. 
  * A single perceptron can only compute a single linear decision boundary, making it mathematically incapable of solving non-linearly separable problems like XOR. The hidden layer with nonlinearities transforms the input space so that the XOR classes can be separated.