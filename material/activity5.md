# Activity 5 (Optional): Visualizing Nonlinear Decision Boundaries

In Activity 1, we discovered that a single perceptron cannot solve XOR because it can only create a straight decision boundary.

In Activity 2, we used an MLP to solve XOR.

Now, let's explore a more interesting problem: **Can an MLP learn to distinguish points inside a circle from points outside it?**

This optional activity demonstrates how hidden layers and nonlinear activation functions allow an MLP to create complex decision boundaries.

## Learning Objectives

By the end of this activity, you should be able to:

1. Generate a synthetic classification dataset.
2. Visualize linearly and nonlinearly separable data.
3. Build an MLP for binary classification.
4. Visualize the decision boundary learned by an MLP.
5. Experiment with network architectures and observe how decision boundaries change.

---

## 1. Generate the Dataset

We will create 400 random points on a two-dimensional plane.

Each point has two coordinates: \(x_1\) and \(x_2\).

We assign a class based on the point's distance from the origin:

- **Class 1:** Inside a circle of radius 1.2.
- **Class 0:** Outside the circle.

Mathematically, a point is inside the circle when:

\[
x_1^2 + x_2^2 \leq 1.2^2
\]

### Generate the data

```python
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import Sequential, Input
from keras.layers import Dense

np.random.seed(42)

# Generate 400 random points in 2D
X_circle = np.random.uniform(-2, 2, (400, 2))

# Class 1: inside the circle
# Class 0: outside the circle
y_circle = (
    X_circle[:, 0]**2 + X_circle[:, 1]**2 <= 1.2**2
).astype(int)

print("Input shape:", X_circle.shape)
print("Target shape:", y_circle.shape)
```

### Questions

1. How many features does each example have?
2. What does `np.random.uniform(-2, 2, (400, 2))` generate?
3. Why is this a classification problem rather than regression?

<details>
<summary><strong>Sample Explanation</strong></summary>

Each example has two features: its x-coordinate and y-coordinate.

`np.random.uniform()` generates 400 random points, with coordinates between -2 and 2.

This is a binary classification problem because we predict one of two classes: inside or outside the circle.

</details>

---

## 2. Visualize the Data

Before building a model, let's visualize the classification problem.

### Your task

Create a scatter plot where:

- Points inside the circle are red.
- Points outside the circle are blue.

**Hint:** Use `plt.scatter()` and pass the class labels as the `c` argument.

```python
# YOUR CODE HERE
```

<details>
<summary><strong>Sample Solution</strong></summary>

```python
plt.figure(figsize=(7, 7))

plt.scatter(
    X_circle[:, 0],
    X_circle[:, 1],
    c=y_circle,
    cmap='bwr',
    edgecolor='black'
)

plt.title('Circular Classification Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')
plt.show()
```

</details>

### Questions

1. Can you draw a single straight line that separates the two classes?
2. Could a single perceptron classify all these points correctly?
3. How is this problem similar to XOR?

<details>
<summary><strong>Explanation</strong></summary>

The classes cannot be separated perfectly by a single straight line.

A single perceptron creates a linear decision boundary, so it cannot represent this circular separation exactly.

Like XOR, this is a **nonlinearly separable classification problem**.

An MLP with hidden layers and nonlinear activation functions can represent more complex boundaries.

</details>

---

## 3. Build an MLP

We will create an MLP with:

- 2 input features
- 16 neurons in the first hidden layer
- 16 neurons in the second hidden layer
- 1 output neuron

The hidden layers use ReLU, while the output layer uses sigmoid.

### Your task

Build the model using Keras.

```python
# YOUR CODE HERE
```

<details>
<summary><strong>Sample Solution</strong></summary>

```python
circle_model = Sequential([
    Input(shape=(2,)),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

circle_model.summary()
```

</details>

### Questions

1. Why does the input layer expect two values?
2. Why do the hidden layers use ReLU?
3. Why do we use sigmoid in the output layer?

<details>
<summary><strong>Explanation</strong></summary>

The input layer receives two coordinates for each point.

ReLU introduces nonlinear transformations, allowing the MLP to represent complex boundaries.

Sigmoid produces an output between 0 and 1 that we can interpret as a score for binary classification.

</details>

---

## 4. Compile and Train the Model

We will train the model using the same Keras workflow introduced in Activity 2.

We are not yet studying how optimization and backpropagation work internally. For now, observe what the model learns.

### Your task

Compile the model using:

- Adam optimizer
- Binary cross-entropy loss
- Accuracy metric

Train the model for 300 epochs with a batch size of 16.

```python
# YOUR CODE HERE
```

<details>
<summary><strong>Sample Solution</strong></summary>

```python
circle_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_circle = circle_model.fit(
    X_circle,
    y_circle,
    epochs=300,
    batch_size=16,
    verbose=0
)

print(
    f"Training accuracy: "
    f"{history_circle.history['accuracy'][-1]:.2%}"
)
```

</details>

**Question:** Did the model achieve high training accuracy?

> **Note:** Results may vary because model parameters are initialized randomly. In this activity, we are interested in visualizing the boundary, not achieving a particular accuracy.

---

## 5. Visualize the Decision Boundary

Now comes the most interesting part.

We want to visualize how the MLP classifies different locations on the two-dimensional plane.

We will:

1. Create a grid of coordinates.
2. Ask the model to predict a score for each point.
3. Color the plane according to the predictions.
4. Display the original data points on top.

### Create a grid

```python
xx, yy = np.meshgrid(
    np.linspace(-2.5, 2.5, 100),
    np.linspace(-2.5, 2.5, 100)
)

grid_coordinates = np.c_[
    xx.ravel(),
    yy.ravel()
]

print("Grid shape:", grid_coordinates.shape)
```

**Question:** Why does the grid contain 10,000 coordinate pairs?

<details>
<summary><strong>Explanation</strong></summary>

We create 100 positions on the horizontal axis and 100 positions on the vertical axis.

Therefore:

100 × 100 = 10,000 coordinate pairs.

We use them to inspect the model's predictions across the entire plane.

</details>

### Generate predictions

```python
predictions = circle_model.predict(
    grid_coordinates,
    verbose=0
)

zz = predictions.reshape(xx.shape)
```

### Plot the decision boundary

```python
plt.figure(figsize=(7, 7))

plt.contourf(
    xx,
    yy,
    zz,
    levels=np.linspace(0, 1, 21),
    alpha=0.4,
    cmap='bwr'
)

plt.contour(
    xx,
    yy,
    zz,
    levels=[0.5],
    colors='black',
    linewidths=2
)

plt.scatter(
    X_circle[:, 0],
    X_circle[:, 1],
    c=y_circle,
    cmap='bwr',
    edgecolor='black',
    s=25
)

plt.title('MLP Nonlinear Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')
plt.show()
```

### Observe the result

The background colors show how the MLP classifies different regions.

The black contour indicates the boundary where the model's predicted score is 0.5.

### Questions

1. Does the decision boundary resemble a circle?
2. Is the boundary perfectly smooth?
3. Can you identify regions where the model makes incorrect predictions?
4. Why can an MLP create this boundary while a single perceptron cannot?

<details>
<summary><strong>Explanation</strong></summary>

The MLP can learn a boundary that approximately follows the circular pattern.

With ReLU activations, the network combines multiple piecewise-linear transformations. The resulting decision boundary can resemble a polygon approximating a circle.

A single perceptron can only create a straight boundary.

Hidden layers and nonlinear activation functions allow an MLP to represent more complex patterns.

</details>

---

## 6. Experiment with the Architecture

Let's investigate how the network architecture affects the decision boundary.

### Your challenge

Choose one or more experiments:

1. Remove one hidden layer.
2. Change the number of neurons from 16 to 4.
3. Increase the number of neurons from 16 to 32.
4. Replace ReLU with Tanh.
5. Reduce the number of training epochs.

For each experiment:

- Rebuild the model.
- Compile and train it.
- Generate predictions for the grid.
- Plot the new decision boundary.
- Compare it with the original.

### Record your observations

| Model | Architecture | Activation | Observation |
|---|---|---|---|
| Original | 16 → 16 → 1 | ReLU | |
| Experiment 1 | | | |
| Experiment 2 | | | |

### Discussion questions

- How does reducing the number of neurons affect the boundary?
- What happens when you add more neurons?
- Does a more complex architecture always produce a better boundary?
- What differences do you notice between ReLU and Tanh?

<details>
<summary><strong>Instructor Notes</strong></summary>

The goal is to help students visualize the relationship between network architecture and representational capacity.

Possible observations:

- Smaller models may produce simpler, less accurate boundaries.
- Larger models may approximate the circular boundary more closely.
- ReLU networks often produce polygon-like boundaries.
- Tanh networks can produce smoother-looking boundaries.

Results depend on random initialization and training.

Avoid introducing detailed optimization theory here. Those topics will be covered later in the course.

</details>

---

## Conclusion

In this optional activity, you explored how an MLP learns to classify points using a nonlinear decision boundary.

You discovered that:

- A single perceptron is limited to linear decision boundaries.
- MLPs use hidden layers and nonlinear activation functions to represent more complex relationships.
- Decision boundaries can be visualized by asking a model to make predictions across a grid.
- Changing the architecture can change the boundary learned by the model.

**Final reflection:** How does this activity help explain why hidden layers and nonlinear activation functions are important in neural networks?

