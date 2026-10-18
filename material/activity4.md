# Activity 4: The Concrete Dataset Challenge

In Activity 3, we built a Multi-Layer Perceptron (MLP) using Keras to predict cereal ratings.

Now it's your turn to apply the same workflow to a different regression problem.

**The challenge:** Build an MLP that predicts the compressive strength of concrete based on its ingredients and other input measurements.

This activity is more independent than Activity 3. You will receive instructions, hints, and sample solutions, but you should attempt each task before opening the solutions.

## Learning Objectives

By the end of this activity, you should be able to:

1. Load and explore a dataset.
2. Separate input features from the target variable.
3. Split data into training, validation, and testing sets.
4. Scale numerical features correctly.
5. Build and train an MLP using Keras.
6. Evaluate the model and interpret its predictions.
7. Experiment with different neural network architectures.

---

## 1. Load and Explore the Dataset

The concrete dataset contains information about concrete mixtures and their compressive strength.

Our goal is to predict **CompressiveStrength**, a continuous numerical value, from the input features.

### Load the dataset

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras import Sequential, Input
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

concrete_data = pd.read_csv(
    "https://raw.githubusercontent.com/AI-Learning-Repo/resources/refs/heads/main/datasets/concrete.csv"
)
```

### Your tasks

1. Display the first five rows.
2. Find the number of rows and columns.
3. Check for missing values.
4. Identify the target column.
5. Count the number of input features.

```python
# YOUR CODE HERE

```

<details>
<summary><strong>Sample Solution: Exploring the Dataset</strong></summary>

```python
print(concrete_data.head())

print("Dataset shape:", concrete_data.shape)

print("Missing values:")
print(concrete_data.isnull().sum())

print("Columns:")
print(concrete_data.columns.tolist())
```

The target column is `CompressiveStrength`.

All remaining columns are candidate input features.

</details>

**Question:** Is this a classification or a regression problem? Why?

<details>
<summary><strong>Explanation</strong></summary>

This is a **regression problem** because the target is a continuous numerical value representing concrete compressive strength.

In classification, we predict categories. In regression, we predict numerical values.

</details>

---

## 2. Prepare the Features and Target

Separate the dataset into:

- **X:** Input features used for prediction.
- **y:** Target values we want to predict.

### Your task

Create two variables:

- `concrete_features`
- `concrete_target`

**Hint:** Use `.drop()` to remove the target column from the input features.

```python
# YOUR CODE HERE

```

<details>
<summary><strong>Sample Solution</strong></summary>

```python
concrete_features = concrete_data.drop(
    columns=['CompressiveStrength']
)

concrete_target = concrete_data['CompressiveStrength']

print("Features:", concrete_features.shape)
print("Target:", concrete_target.shape)
```

</details>

**Question:** Why must the target column be removed from the input features?

<details>
<summary><strong>Explanation</strong></summary>

The target is the answer that the model is supposed to predict.

Including it in the input features would give the model access to the correct answer, resulting in **data leakage** and misleading evaluation results.

</details>

---

## 3. Split the Dataset

As in Activity 3, we will create three datasets:

- **Training:** Used to train the model.
- **Validation:** Used to monitor performance and compare model configurations.
- **Testing:** Reserved for final evaluation.

### Your task

Use `train_test_split()` to divide the data.

Use the same approach as Activity 3:

1. Reserve 20% for testing.
2. Split the remaining data into training and validation sets.
3. Use `random_state=32` for reproducibility.

```python
# YOUR CODE HERE

```

<details>
<summary><strong>Sample Solution</strong></summary>

```python
X_train_val, X_test, y_train_val, y_test = train_test_split(
    concrete_features,
    concrete_target,
    test_size=0.2,
    random_state=32
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.2,
    random_state=32
)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)
```

The approximate proportions are:

- Training: 64%
- Validation: 16%
- Testing: 20%

</details>

**Question:** Why shouldn't we repeatedly evaluate different model architectures on the test set?

<details>
<summary><strong>Explanation</strong></summary>

The test set should provide an independent assessment of our final model.

Repeatedly using it to select architectures or settings can make the evaluation overly optimistic.

Use the validation set for experimentation and reserve the test set for the final assessment.

</details>

---

## 4. Scale the Input Features

Neural networks often perform better when input features have similar numerical scales.

For example, different concrete ingredients may be measured in quantities with very different numerical ranges.

We will use `StandardScaler`.

### Your task

Scale the training, validation, and testing features.

**Hint:**

```python
scaler = StandardScaler()

# Learn scaling parameters from training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same scaling to other datasets
# YOUR CODE HERE
```

<details>
<summary><strong>Sample Solution</strong></summary>

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

`fit_transform()` calculates scaling parameters using only the training data.

`transform()` applies those same parameters to validation and testing data.

This prevents information from the other datasets from influencing preprocessing.

</details>

**Question:** What could happen if we used `fit_transform()` separately on the test dataset?

---

## 5. Build Your MLP

Now let's build a neural network to predict concrete compressive strength.

You can use the cereal model from Activity 3 as a starting point.

### Your task

Create a Keras `Sequential` model with:

- An input layer matching the number of features.
- A first hidden layer containing 128 neurons.
- A second hidden layer containing 64 neurons.
- ReLU activation functions in the hidden layers.
- One output neuron for regression.

Use `model.summary()` to inspect the architecture.

```python
# YOUR CODE HERE

```

<details>
<summary><strong>Sample Solution</strong></summary>

```python
concrete_model = Sequential()

concrete_model.add(
    Input(shape=(X_train_scaled.shape[1],))
)

concrete_model.add(Dense(128, activation='relu'))
concrete_model.add(Dense(64, activation='relu'))

concrete_model.add(Dense(1))

concrete_model.summary()
```

The output layer contains one neuron because we are predicting one continuous numerical value.

The output layer uses the default linear activation function, which is suitable for regression.

</details>

### Questions

1. How many input features does the model receive?
2. Why do the hidden layers use nonlinear activation functions?
3. Why is the output layer different from the one we used for XOR?
4. How many trainable parameters does your model have?

<details>
<summary><strong>Explanation</strong></summary>

The number of input features is determined by the dataset.

ReLU introduces nonlinearity, allowing the network to model complex relationships.

For XOR, we used a sigmoid output because we were performing binary classification.

For concrete strength, we use a linear output because we are predicting a continuous numerical value.

You can calculate trainable parameters using:

**Parameters = (number of inputs × number of neurons) + number of biases**

Compare your calculation with `model.summary()`.

</details>

---

## 6. Compile the Model

Prepare the model for training.

### Your task

Use:

- `adam` optimizer
- `mean_squared_error` loss
- `mae` metric

```python
# YOUR CODE HERE

```

<details>
<summary><strong>Sample Solution</strong></summary>

```python
concrete_model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)
```

- **Optimizer:** Controls how model parameters are updated.
- **Loss:** Measures the error being minimized during training.
- **Metric:** Reports model performance in an understandable way.

We will explore their underlying mathematics in future sessions.

</details>

**Question:** Why do we use MSE and MAE for this problem instead of binary cross-entropy and classification accuracy?

---

## 7. Train the Model

Train your model using the training data.

Use the validation data to observe how well the model performs on examples not used to update the weights.

### Your task

Train the model using:

- `epochs=100`
- `batch_size=32`
- The scaled training features
- The validation dataset

Store the training history in a variable named `history_concrete`.

```python
# YOUR CODE HERE

```

<details>
<summary><strong>Sample Solution</strong></summary>

```python
history_concrete = concrete_model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    verbose=0
)

print("Training complete.")
```

An epoch represents one pass through the training dataset.

Batch size determines how many examples are processed together during each training step.

The validation data is used to monitor performance after each epoch.

</details>

### Questions

1. What happens if you increase the number of epochs?
2. Does more training always guarantee better predictions?
3. Why do we provide validation data?

We will study the training process and techniques for improving it in more detail next week.

---

## 8. Visualize Training Results

The `history_concrete` object contains information collected during training.

### Your task

Plot:

- Training loss
- Validation loss

Use the epoch number on the horizontal axis.

```python
# YOUR CODE HERE

```

<details>
<summary><strong>Sample Solution</strong></summary>

```python
plt.plot(
    history_concrete.history['loss'],
    label='Training Loss'
)

plt.plot(
    history_concrete.history['val_loss'],
    label='Validation Loss'
)

plt.title('Concrete Strength Prediction')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
```

</details>

### Discussion questions

1. Does training loss decrease?
2. Does validation loss decrease?
3. Are the two curves similar?
4. What might it mean if validation loss starts increasing while training loss continues decreasing?

<details>
<summary><strong>Explanation</strong></summary>

Training loss measures performance on the training dataset.

Validation loss measures performance on data that is not used to update the model's weights.

If training loss continues decreasing while validation loss increases, this may indicate **overfitting**.

We will explore ways to manage overfitting, including early stopping, in later sessions.

</details>

---

## 9. Evaluate the Model

Now evaluate the model on the test data.

The test dataset has not been used to train the model or select its architecture.

### Your task

Use `model.evaluate()` to calculate:

- Test Mean Squared Error (MSE)
- Test Mean Absolute Error (MAE)

```python
# YOUR CODE HERE

```

<details>
<summary><strong>Sample Solution</strong></summary>

```python
test_loss, test_mae = concrete_model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print(f"Test MSE: {test_loss:.2f}")
print(f"Test MAE: {test_mae:.2f}")
```

</details>

**Question:** What does the test MAE tell us about the model's prediction quality?

<details>
<summary><strong>Explanation</strong></summary>

MAE represents the average absolute difference between predicted and actual concrete strength values.

For example, if the MAE is 5, predictions differ from actual values by approximately 5 units of compressive strength on average.

A smaller MAE generally indicates better predictions on the same dataset.

</details>

---

## 10. Make and Inspect Predictions

Let's compare the predicted concrete strengths with the actual values.

### Your task

1. Generate predictions for the test dataset.
2. Create a DataFrame containing actual and predicted values.
3. Calculate the absolute error.
4. Display the first 10 rows.

```python
# YOUR CODE HERE

```

<details>
<summary><strong>Sample Solution</strong></summary>

```python
predictions = concrete_model.predict(
    X_test_scaled,
    verbose=0
).flatten()

results = pd.DataFrame({
    'Actual': y_test.to_numpy(),
    'Predicted': predictions
})

results['Absolute Error'] = abs(
    results['Actual'] - results['Predicted']
)

print(results.head(10))
```

</details>

### Questions

1. Which prediction has the smallest error?
2. Which prediction has the largest error?
3. Does the model predict all examples equally well?
4. What factors might influence the prediction errors?

---

## 11. Experiment with the Architecture

You now have a complete working regression model.

Let's explore how the architecture affects its performance.

### Your challenge

Choose **one or two modifications**:

1. Use only one hidden layer.
2. Change the hidden layer sizes from `128 → 64` to `64 → 32`.
3. Increase the number of neurons.
4. Replace ReLU with Tanh.
5. Change the number of epochs from 100 to 50 or 150.

**Important:** Build a new model for each architecture instead of continuing to train the previous model.

Use the validation set for comparisons, and evaluate on the test set only after choosing your final model.

Record your observations:

| Model | Architecture | Epochs | Final validation MAE |
|---|---|---|---|
| Original | 128 → 64 → 1 | 100 | |
| Modified 1 | | | |
| Modified 2 | | | |

### Discussion questions

- Which model achieved the lowest validation MAE?
- Did adding more neurons always improve the results?
- Did longer training necessarily produce better validation results?
- What conclusions can you draw from your experiments?

<details>
<summary><strong>Instructor Notes: Experimentation</strong></summary>

The purpose is to encourage experimentation, not to find a universally optimal model.

Students should observe that architectural decisions and training settings influence performance.

Because weights are initialized randomly, results can differ between training runs.

A single experiment is not sufficient to establish that one architecture is always better.

We will explore model optimization and more systematic evaluation techniques as the course progresses.

</details>

---

## Conclusion

In this activity, you independently built an MLP to predict concrete compressive strength.

You practiced the complete workflow:

1. Loading and exploring data.
2. Preparing features and targets.
3. Splitting and scaling data.
4. Building and training an MLP.
5. Visualizing training results.
6. Evaluating predictions.
7. Experimenting with neural network architectures.

**Reflection question:** What are the main differences between the MLP used for XOR classification and the MLP used for concrete strength regression?

<details>
<summary><strong>Sample Discussion</strong></summary>

Both models use the same basic architecture: input features, hidden layers, and an output layer.

However:

- XOR is a **binary classification problem**, while concrete strength prediction is a **regression problem**.
- XOR uses a sigmoid output; concrete strength prediction uses a linear output.
- XOR uses binary cross-entropy and accuracy; concrete strength prediction uses MSE and MAE.
- Concrete strength prediction uses more input features and a larger dataset.

The fundamental Keras workflow is similar for both tasks.

</details>