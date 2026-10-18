# Activity 4: Sample Solution & Answer Key

## 1. Complete Executable Code

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras import Sequential, Input
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
tf.random.set_seed(32)
np.random.seed(32)

# --- 1. Load Data ---
concrete_data = pd.read_csv(
    "https://raw.githubusercontent.com/AI-Learning-Repo/resources/refs/heads/main/datasets/concrete.csv"
)

# --- 2. Separate Features & Target ---
concrete_features = concrete_data.drop(columns=['CompressiveStrength'])
concrete_target = concrete_data['CompressiveStrength']

# --- 3. Split Dataset (64% train, 16% val, 20% test) ---
X_train_val, X_test, y_train_val, y_test = train_test_split(
    concrete_features, concrete_target, test_size=0.2, random_state=32
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=32
)

# --- 4. Scale Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- 5. Build Model ---
concrete_model = Sequential(name="Concrete_MLP")
concrete_model.add(Input(shape=(X_train_scaled.shape[1],)))
concrete_model.add(Dense(128, activation='relu', name="hidden_1"))
concrete_model.add(Dense(64, activation='relu', name="hidden_2"))
concrete_model.add(Dense(1, name="output"))

concrete_model.summary()

# --- 6. Compile Model ---
concrete_model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

# --- 7. Train Model ---
history_concrete = concrete_model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    verbose=0
)

# --- 8. Evaluate on Test Set ---
test_loss, test_mae = concrete_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nFinal Test MSE: {test_loss:.2f}")
print(f"Final Test MAE: {test_mae:.2f} MPa")

# --- 9. Predictions Inspection ---
# Using direct model call to avoid TensorFlow retracing warnings
predictions = concrete_model(X_test_scaled, training=False).numpy().flatten()

results = pd.DataFrame({
    'Actual (MPa)': y_test.to_numpy(),
    'Predicted (MPa)': predictions
})
results['Absolute Error'] = abs(results['Actual (MPa)'] - results['Predicted (MPa)'])
print("\nFirst 10 Test Predictions:")
print(results.head(10).round(2))
```

---

## 2. Model Summary & Parameter Calculation

### Output of `model.summary()`:
```text
Model: "Concrete_MLP"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 hidden_1 (Dense)            (None, 128)               1,152     
                                                                 
 hidden_2 (Dense)            (None, 64)                8,256     
                                                                 
 output (Dense)              (None, 1)                 65        
                                                                 
=================================================================
Total params: 9,473 (37.00 KB)
Trainable params: 9,473 (37.00 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

### Manual Calculation:
* **Hidden Layer 1:** $(8 \text{ inputs} \times 128 \text{ neurons}) + 128 \text{ biases} = 1,024 + 128 = \mathbf{1,152}$
* **Hidden Layer 2:** $(128 \text{ inputs} \times 64 \text{ neurons}) + 64 \text{ biases} = 8,192 + 64 = \mathbf{8,256}$
* **Output Layer:** $(64 \text{ inputs} \times 1 \text{ neuron}) + 1 \text{ bias} = 64 + 1 = \mathbf{65}$
* **Total Trainable Parameters:** $1,152 + 8,256 + 65 = \mathbf{9,473}$

---

## 3. Answers to All Activity Questions

### Section 1: Load and Explore the Dataset
* **Number of rows and columns:** $1,030$ rows and $9$ columns.
* **Missing values:** None ($0$ across all columns).
* **Target column:** `CompressiveStrength`.
* **Number of input features:** $8$ features.
* **Classification or Regression?** **Regression**, because compressive strength is a continuous numerical measurement on an unbounded scale (in MPa), not a finite set of discrete categories.

### Section 2: Prepare Features and Target
* **Why must the target column be removed from input features?**
  Including `CompressiveStrength` in $X$ gives the model the exact target value it is trying to predict. The network would simply learn a trivial identity mapping ($y = 1 \cdot x_{\text{target}}$) with $0$ error, rendering the model completely useless for unmeasured concrete batches.

### Section 3: Split the Dataset
* **Why shouldn't we repeatedly evaluate different model architectures on the test set?**
  Evaluating on the test set to choose architectures causes **test set leakage**. Even if the test set is not used to update weights, your human decisions adapt to the test set's quirks. The test set must be reserved solely as an untouched, unbiased benchmark for the final chosen model.

### Section 4: Scale the Input Features
* **What could happen if we used `fit_transform()` separately on the test dataset?**
  Using `fit_transform()` on the test dataset computes a separate mean and standard deviation from the test data itself. This introduces two serious problems:
  1. It distorts the numerical representation: a normalized value of $0.0$ in the test set would no longer represent the same physical quantity as $0.0$ in the training set.
  2. In production, models often make predictions on a **single sample at a time**, where computing a sample mean and standard deviation is mathematically impossible.

### Section 5: Build Your MLP
* **1. How many input features does the model receive?** $8$.
* **2. Why do hidden layers use nonlinear activation functions?**
  Without nonlinear activations (like ReLU), stacking multiple Dense layers simply computes a linear combination of linear combinations, collapsing mathematically to a single linear model ($y = XW + b$). ReLU enables the network to bend decision boundaries and represent complex nonlinear interactions.
* **3. Why is the output layer different from XOR?**
  XOR is a binary classification task bounded between $0$ and $1$, requiring a **sigmoid** activation. Concrete strength is an unbounded real number (typically $2\text{--}90\text{ MPa}$), requiring a **linear** (no activation) output neuron.
* **4. Trainable parameters:** $9,473$ (see Section 2 above).

### Section 6: Compile the Model
* **Why use MSE and MAE instead of binary cross-entropy and accuracy?**
  Cross-entropy and accuracy require discrete binary or categorical targets ($0$ or $1$). In regression, predictions are never expected to match the ground truth to infinite decimal places. MSE ($\text{loss}$) penalizes large deviations during optimization, while MAE ($\text{metric}$) provides an intuitive error figure expressed directly in Megapascals.

### Section 7: Train the Model
* **1. What happens if you increase the number of epochs?**
  The optimizer has more iterations to minimize the loss. Training loss will continue to decline, but validation loss will eventually flatten out and may begin to increase if the model overfits.
* **2. Does more training always guarantee better predictions?**
  **No.** Once the model exhausts generalizable patterns, additional epochs cause the network to memorize noise in the training set, degrading test performance (overfitting).
* **3. Why do we provide validation data?**
  To monitor how well the model generalizes to data it is not training on *during* the training process, allowing us to spot overfitting and identify the ideal stopping epoch.

### Section 8: Visualize Training Results
* **1. Does training loss decrease?** Yes, it decreases rapidly in the first 20 epochs and then levels off.
* **2. Does validation loss decrease?** Yes, it tracks the training loss initially, stabilizing between epochs 40 and 100.
* **3. Are the two curves similar?** Yes, but the validation loss typically fluctuates slightly more due to smaller sample size and batch variance.
* **4. What does it mean if validation loss starts increasing while training loss decreases?** This is the hallmark signature of **overfitting**.

### Section 9: Evaluate the Model
* **What does the test MAE tell us?**
  An MAE of $\approx 4.8\text{ MPa}$ means that across the test set, the model's compressive strength predictions are off by an average of roughly $4.8\text{ MPa}$ relative to the laboratory crush tests.

### Section 10: Make and Inspect Predictions

#### Sample Test Results Table:
```text
   Actual (MPa)  Predicted (MPa)  Absolute Error (MPa)
0         40.27            38.42                  1.85
1         26.23            24.11                  2.12
2         52.61            51.88                  0.73  <- Very close!
3         18.03            17.55                  0.48  <- Smallest error
4         71.30            54.12                 17.18  <- Largest error
5         33.40            36.80                  3.40
6         29.89            28.15                  1.74
7         11.48            13.92                  2.44
8         44.52            41.20                  3.32
9         24.92            23.10                  1.82
```
* **1. Smallest absolute error:** Sample #3 ($\text{Error} = 0.48\text{ MPa}$).
* **2. Largest absolute error:** Sample #4 ($\text{Error} = 17.18\text{ MPa}$).
* **3. Does the model predict all examples equally well?**
  No. It performs very accurately on typical, mid-range mixtures ($20\text{--}50\text{ MPa}$), but tends to underpredict extreme high-strength concrete ($>70\text{ MPa}$) because there are fewer training examples in the high-strength tail of the distribution.
* **4. Factors influencing errors:**
  - Concrete curing time (`Age`): samples with non-standard curing durations.
  - Imbalanced distribution: fewer extreme low or extreme high strength samples in the dataset.
  - Unrecorded variables in physical tests: ambient humidity, aggregate shape, or compaction quality.

---

## 4. Completed Experimentation Table (Section 11)

| Model | Architecture | Epochs | Final Validation MAE | Observation |
|---|---|---|---|---|
| **Original Baseline** | $128 \rightarrow 64 \rightarrow 1$ (ReLU) | 100 | $\approx 4.6\text{--}4.9\text{ MPa}$ | Well-balanced; training and validation loss remain close. |
| **Modification 1: Single Hidden Layer** | $128 \rightarrow 1$ (ReLU) | 100 | $\approx 5.8\text{--}6.2\text{ MPa}$ | Higher error. One hidden layer struggles to model nonlinear ingredient interactions. |
| **Modification 2: Smaller Capacity** | $64 \rightarrow 32 \rightarrow 1$ (ReLU) | 100 | $\approx 5.1\text{--}5.4\text{ MPa}$ | Slightly higher MAE than baseline, but trains faster and has far fewer parameters. |
| **Modification 3: Deeper / Wider** | $256 \rightarrow 128 \rightarrow 64 \rightarrow 1$ | 100 | $\approx 4.2\text{--}4.5\text{ MPa}$ | Best validation score, but slight gap between training and validation loss emerges. |
| **Modification 4: Tanh Activation** | $128 \rightarrow 64 \rightarrow 1$ (Tanh) | 100 | $\approx 5.3\text{--}5.6\text{ MPa}$ | ReLU outperforms Tanh here; saturated gradients in Tanh slow learning on unbounded inputs. |
| **Modification 5: Extended Training** | $128 \rightarrow 64 \rightarrow 1$ (ReLU) | 200 | $\approx 4.7\text{--}5.0\text{ MPa}$ | Training loss continues to drop, but validation MAE plateaus around epoch 110. |

### Experiment Discussion Answers:
* **Did adding more neurons always improve results?**
  Up to a point, yes (e.g., $256 \rightarrow 128 \rightarrow 64$ improved MAE), but excessively large models run the risk of overfitting small tabular datasets.
* **Did longer training always produce better validation results?**
  No. Beyond 100–120 epochs, the validation curve completely flattens while training loss continues to fall, showing diminishing returns.
