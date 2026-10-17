# Activity 1: Perceptrons, Logic Gates, and the XOR Limitation

In this activity, you will build a **single perceptron**, explore how its weights and bias affect its output, and discover why a single perceptron cannot solve every classification problem. This prepares us for **Multi-Layer Perceptrons (MLPs)** in Activity 2.

**Learning objectives**

1. Implement a perceptron with a step activation function in Python.
2. Experiment with weights and biases to model OR, AND, and NOT logic gates.
3. Visualize the difference between linearly separable and non-linearly separable data.
4. Explain, in your own words, why a single perceptron cannot solve XOR.

**How to work:** Complete the tasks before opening the sample solutions. Work with a partner when possible: predict the outcome first, run the code, then explain any difference.

---

## The Perceptron - Computational Capacity and Limits

The Perceptron acts as a linear classifier. It computes the dot product of input vectors and weight vectors, adds a bias scalar, and applies a Heaviside step function to yield a binary output.

### 1 Implementing the Perceptron

The following Python function models the Perceptron utilizing NumPy for optimized vector operations.

```python
import numpy as np
import matplotlib.pyplot as plt

def perceptron(inputs, weights, bias):
    """Calculates the output of a single perceptron utilizing a step activation function."""
    assert len(inputs) == len(weights), "Dimensionality mismatch between inputs and weights."
    
    # Calculate the weighted sum using the dot product
    weighted_sum = np.dot(inputs, weights) + bias

    # --- YOUR CODE HERE --- #
    # Apply the step activation function:
    # If weighted_sum is greater than or equal to 0, output 1
    # Otherwise, output 0
    # --- END YOUR CODE --- #

# Test functionality
test_inputs = [1, 0]
test_weights = [0.5, 0.5]
test_bias = -0.7
print(f"Test Output: {perceptron(test_inputs, test_weights, test_bias)}") 
```

<details>
<summary><b>Sample Solution & Code Explanation: Perceptron Function</b></summary>

**Solution:**
```python
def perceptron(inputs, weights, bias):
    weighted_sum = np.dot(inputs, weights) + bias
    if weighted_sum >= 0:
        return 1
    else:
        return 0
```

**Code explanation:**
*   `import numpy as np`: Imports the NumPy library, which handles numerical arrays and matrices efficiently.
*   `assert len(inputs) == len(weights)`: A safety check. It ensures we have exactly one weight for every input. If they don't match, the program throws an error.
*   `np.dot(inputs, weights)`: This performs the "dot product". Instead of writing a `for` loop to multiply `input[0] * weight[0] + input[1] * weight[1]...`, NumPy calculates this entire sum mathematically at the hardware level.
*   `if weighted_sum >= 0:`: This is the Heaviside step function. It acts as the activation threshold, converting the continuous numerical sum into a discrete binary output (1 or 0).
</details>

**Try it:** Before running the test, calculate the weighted sum by hand. What output do you expect? Then change only `test_bias` to `-0.3`. Predict the new output and explain what changed.

### 2 Modeling Logic Gates

By modifying the weights and bias, the perceptron can model linearly separable Boolean logic.

**Task:** Choose weights and biases for each gate. First predict all outputs from its truth table, then test your choices. There is more than one correct set of weights and biases.

**(a) OR Gate:** Outputs 1 if at least one input is 1.
```python
gate_inputs = [[0, 0], [0, 1],[1, 0], [1, 1]]

# --- YOUR TASK: Define weights and bias for OR --- #
weights_or = [None, None]
bias_or = None
# --- END YOUR TASK --- #

print("--- OR Gate Output ---")
for input_pair in gate_inputs:
    output = perceptron(input_pair, weights_or, bias_or)
    print(f"Input: {input_pair}, Output: {output}")
```

<details>
<summary><b>Sample Solution & Explanation: OR Gate</b></summary>

**Solution:**
```python
weights_or = [1.0, 1.0]
bias_or = -0.5
```
**Explanation:** If either input is `1`, the sum is at least `1.0`. Subtracting the bias of `0.5` yields `0.5`, which is $\ge 0$ (fires). If both are `0`, the result is `-0.5` (does not fire).
</details>

**Quick experiment:** Keep the OR weights unchanged and change only its bias. Can you make the same perceptron behave like an AND gate? Why?

**(b) AND Gate:** Outputs 1 strictly when both inputs are 1.
```python
# --- YOUR TASK: Define weights and bias for AND --- #
weights_and = [None, None]
bias_and = None
# --- END YOUR TASK --- #

print("--- AND Gate Output ---")
for input_pair in gate_inputs:
    output = perceptron(input_pair, weights_and, bias_and)
    print(f"Input: {input_pair}, Output: {output}")
```

<details>
<summary><b>Sample Solution & Explanation: AND Gate</b></summary>

**Solution:**
```python
weights_and = [1.0, 1.0]
bias_and = -1.5
```
**Explanation:** By lowering the bias to `-1.5`, a single `1` input yields `1.0 - 1.5 = -0.5` (does not fire). Both inputs must be `1` to yield `2.0 - 1.5 = 0.5` (fires).
</details>


**(c) NOT Gate:** Outputs the inverse of a single input.
```python
not_inputs = [[0], [1]] 

# --- YOUR TASK: Define weight and bias for NOT --- #
weights_not = [None]
bias_not = None
# --- END YOUR TASK --- #

print("\n--- NOT Gate ---")
for input_val in not_inputs:
    # Make sure weights_not matches the input structure (single element)
    output = perceptron(input_val, weights_not, bias_not)
    print(f"Input: {input_val}, Output: {output}")
```
<details>
<summary><b>Sample Solution: NOT Gate</b></summary>

```python
weights_not = [-1.0]
bias_not = 0.5
```
</details>

**Discuss with your partner:** What role does the bias play? What happens if you change the sign of the weight in the NOT gate?

### 3 Visualizing Linear Separability and the XOR Limitation

As established theoretically, a single Perceptron is restricted to defining linear decision boundaries. To understand why the single Perceptron fails, we can visualize the mathematical space. The code below plots the coordinates of the AND gate and the XOR gate on a 2D plane. **Before running it**, sketch both sets of points and try to draw one straight line separating outputs 0 and 1.

```python
# Define the input points as a NumPy array for easier slicing
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
x_coords = inputs[:, 0]
y_coords = inputs[:, 1]

# Define the expected outputs
outputs_and = np.array([0, 0, 0, 1])
outputs_xor = np.array([0, 1, 1, 0])

# Colors map: Index 0 is blue, Index 1 is red
colors = ['blue', 'red']

def plot_gate(gate_outputs, title):
    plt.figure(figsize=(5, 5)) 
    plt.title(title)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    
    # Plot each coordinate with the corresponding color based on its output
    for i in range(len(inputs)):
        plt.scatter(x_coords[i], y_coords[i], color=colors[gate_outputs[i]], s=150, zorder=3)
        
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.grid(True, zorder=0)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # Create legend manually
    import matplotlib.lines as mlines
    blue_marker = mlines.Line2D([],[], color='blue', marker='o', linestyle='None', markersize=10, label='Output 0')
    red_marker = mlines.Line2D([],[], color='red', marker='o', linestyle='None', markersize=10, label='Output 1')
    plt.legend(handles=[blue_marker, red_marker], loc='best')
    
    plt.show()

# Execute plotting
plot_gate(outputs_and, 'AND Gate Outputs (Linearly Separable)')
plot_gate(outputs_xor, 'XOR Gate Outputs (Non-Linearly Separable)')
```

<details>
<summary><b>Code Explanation: Matplotlib Plotting Logic</b></summary>

*   `inputs = np.array(...)`: Converts the nested Python lists into a multi-dimensional NumPy array.
*   `inputs[:, 0]`: This is NumPy slicing syntax. It means "take all rows (`:`), but only the element at index `0` (the first column)". This extracts all the X coordinates into a single list.
*   `plt.figure(figsize=(5, 5))`: Initializes a plotting canvas of 5x5 inches.
*   `color=colors[gate_outputs[i]]`: A clever trick. If the expected output is `0`, it accesses `colors[0]` (blue). If output is `1`, it accesses `colors[1]` (red).
*   `plt.scatter(...)`: Places a dot on the graph at the specified X and Y coordinate. `s=150` controls the size of the dot.
*   `plt.axhline` / `plt.axvline`: Draws the distinct black horizontal and vertical lines representing the X=0 and Y=0 axes.
</details>

> [!IMPORTANT]  
> **The XOR challenge:** Compare the plots. Can you draw one straight line that separates output 0 from output 1 for AND? Can you do the same for XOR? Try changing the perceptron weights and bias to make XOR work. Record what you discover.


**Discuss (required):** Why can a single perceptron solve AND but not XOR? Explain using the plots and the idea of a straight-line decision boundary. You do **not** need to know how networks are trained yet.

**Optional mathematical challenge:** Show why no weights $w_1$, $w_2$ and bias $b$ can reproduce the XOR truth table when the perceptron returns 1 for $w_1x_1+w_2x_2+b \ge 0$.

<details>
<summary><b>Optional solution: Why XOR is impossible for one perceptron</b></summary>

For XOR, all four conditions must hold:

1. Input `(0, 0)` gives 0: $b < 0$.
2. Input `(1, 0)` gives 1: $w_1 + b \ge 0$.
3. Input `(0, 1)` gives 1: $w_2 + b \ge 0$.
4. Input `(1, 1)` gives 0: $w_1 + w_2 + b < 0$.

Adding conditions 2 and 3 gives $w_1 + w_2 + 2b \ge 0$.

Because $b < 0$, we have $w_1 + w_2 + b > w_1 + w_2 + 2b \ge 0$. This contradicts condition 4, which requires $w_1 + w_2 + b < 0$.

Therefore, **no single perceptron can implement XOR**.

</details>

---

## Takeaway: Why Do We Need More Than One Perceptron?

A single perceptron can separate points with **one straight decision boundary**. XOR requires something more flexible.

**Exit question:** What might happen if we connect several artificial neurons in layers instead of using just one?

We will explore that idea in **Activity 2**, where we build an MLP using Keras. For now, focus on the architecture and its behavior; we will study how training works internally later in the course.
