# Activity 1: Perceptrons, Logic Gates, and the XOR Limitation

In the theory lesson, we learned how a perceptron calculates a weighted sum, adds a bias, and applies a step activation. We also saw why a single perceptron can model AND and OR but cannot solve XOR.

In this activity, you will **implement a single perceptron**, choose its weights and bias manually, and use plots to explain its limitations. This prepares you to build a **Multi-Layer Perceptron (MLP)** in Activity 2.

**Learning objectives.** By the end of this activity, you should be able to:

1. Implement a perceptron with a step activation function in Python.
2. Configure OR, AND, and NOT gates using weights and biases.
3. Explain how changes in the bias affect the outputs.
4. Visualize linearly separable and nonlinearly separable binary data.
5. Explain why a single perceptron cannot solve XOR.

**Requirements:** Python 3, NumPy, and Matplotlib (for example, in a Jupyter notebook or Google Colab).

**How to work:** **Predict → Code → Run → Explain.** Complete each task before opening the corresponding sample solution. Run the sections **in order** because later sections reuse earlier variables and functions. Some starter values are intentionally incomplete: fill them in before running that cell.

> [!NOTE]
> You will **choose** weights and biases yourself in this lab. The perceptron is **not trained automatically**. Training an MLP using Keras begins in Activity 2, and the mathematics of training comes later in the course.

---

## 1. Implement the Perceptron

A perceptron computes:

$$z=\mathbf{w}\cdot\mathbf{x}+b$$

and then applies the step function:

$$\operatorname{step}(z)=\begin{cases}1,&z\ge0\\0,&z<0.\end{cases}$$

### Before you code

For `inputs = [1, 0]`, `weights = [0.5, 0.5]`, and `bias = -0.7`:

1. Calculate the weighted sum $z$ by hand.
2. Predict the output (0 or 1).
3. Predict what will happen if **only** the bias changes to `-0.3`.

### Task 1A: Complete the function

```python
import numpy as np
import matplotlib.pyplot as plt


def perceptron(inputs, weights, bias):
    """Return 0 or 1 for a single step-activation perceptron."""
    assert len(inputs) == len(weights), "Inputs and weights must have equal length."
    weighted_sum = np.dot(inputs, weights) + bias

    # YOUR CODE HERE:
    # Return 1 when weighted_sum >= 0; otherwise return 0.
    raise NotImplementedError("Complete the step activation before running the tests.")
```

> [!TIP]
> Replace the `raise NotImplementedError(...)` line with your own `if/else` statement or a one-line conditional expression.

### Task 1B: Test your implementation

After completing the function, run:

```python
print("First output:", perceptron([1, 0], [0.5, 0.5], -0.7))
print("Changed bias:", perceptron([1, 0], [0.5, 0.5], -0.3))

assert perceptron([1, 0], [0.5, 0.5], -0.7) == 0
assert perceptron([1, 0], [0.5, 0.5], -0.3) == 1
print("Both tests passed!")
```

**Question:** The inputs and weights did not change. Why did changing only the bias change the output?

<details>
<summary><strong>Sample solution and explanation: Section 1</strong></summary>

```python
def perceptron(inputs, weights, bias):
    """Return 0 or 1 for a single step-activation perceptron."""
    assert len(inputs) == len(weights), "Inputs and weights must have equal length."
    weighted_sum = np.dot(inputs, weights) + bias

    if weighted_sum >= 0:
        return 1
    else:
        return 0
```

With bias `-0.7`, $z=(1)(0.5)+(0)(0.5)-0.7=-0.2$ and the output is **0**. With bias `-0.3`, $z=0.2$ and the output is **1**. Changing the bias moves the activation threshold relative to the input's weighted sum.

- `np.dot(inputs, weights)` calculates the weighted sum without the bias.
- The `assert` checks that there is one weight per input.
- `weighted_sum >= 0` determines the output class.

</details>

---

## 2. Model Logic Gates

A logic gate accepts binary inputs and produces a binary output. Your job is to choose parameters that reproduce the target truth table.

**Important:** There are **many** correct choices of weights and bias. The goal is to explain **why** your choice works, not memorize one answer.

### Task 2A: OR gate

OR outputs 1 when at least one of its two inputs is 1.

| Input 1 | Input 2 | Expected OR |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

1. Choose two weights and one bias.
2. Predict all four outputs before running the code.
3. Run the cell and check that your solution reproduces the truth table.

```python
gate_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
expected_or = [0, 1, 1, 1]

# REPLACE ALL None VALUES BEFORE RUNNING
weights_or = [None, None]
bias_or = None

actual_or = [perceptron(pair, weights_or, bias_or) for pair in gate_inputs]

for pair, actual, expected in zip(gate_inputs, actual_or, expected_or):
    print(f"Input: {pair} | Predicted: {actual} | Expected: {expected}")

assert actual_or == expected_or, "Your OR gate does not yet match its truth table."
print("OR gate works!")
```

<details>
<summary><strong>Sample solution: OR</strong></summary>

```python
weights_or = [1.0, 1.0]
bias_or = -0.5
```

This gives $z=x_1+x_2-0.5$. For $(0,0)$, $z=-0.5$ and the output is 0. If either input is 1, $z\ge0.5$ and the output is 1.

Other correct configurations are possible. For example, multiplying both the weights **and** bias by the same positive constant preserves the decisions.

</details>

### Task 2B: AND gate — change the bias

AND outputs 1 **only when both** inputs are 1.

| Input 1 | Input 2 | Expected AND |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

**First try:** Keep the OR weights unchanged. Change **only the bias** so the perceptron behaves like AND.

```python
expected_and = [0, 0, 0, 1]

# Try copying your OR weights and changing the bias.
weights_and = [None, None]  # REPLACE
bias_and = None             # REPLACE

actual_and = [perceptron(pair, weights_and, bias_and) for pair in gate_inputs]

for pair, actual, expected in zip(gate_inputs, actual_and, expected_and):
    print(f"Input: {pair} | Predicted: {actual} | Expected: {expected}")

assert actual_and == expected_and, "Your AND gate does not yet match its truth table."
print("AND gate works!")
```

**Questions:** Why does one input equal to 1 produce output 0, but two inputs equal to 1 produce output 1? What did changing the bias accomplish?

<details>
<summary><strong>Sample solution: AND</strong></summary>

```python
weights_and = [1.0, 1.0]
bias_and = -1.5
```

Now $z=x_1+x_2-1.5$. When only one input is 1, $z=-0.5$, so the output is 0. Only $(1,1)$ gives $z=0.5$ and output 1.

This shows how adjusting the bias changes where the decision boundary lies.

</details>

### Task 2C: NOT gate

NOT takes **one** input, rather than two, and returns the opposite binary value.

| Input | Expected NOT |
|:---:|:---:|
| 0 | 1 |
| 1 | 0 |

**Hint:** Use one negative weight and a suitable positive bias.

```python
not_inputs = [[0], [1]]
expected_not = [1, 0]

# REPLACE ALL None VALUES BEFORE RUNNING
weights_not = [None]
bias_not = None

actual_not = [perceptron(value, weights_not, bias_not) for value in not_inputs]

for value, actual, expected in zip(not_inputs, actual_not, expected_not):
    print(f"Input: {value} | Predicted: {actual} | Expected: {expected}")

assert actual_not == expected_not, "Your NOT gate does not yet match its truth table."
print("NOT gate works!")
```

**Question:** Why does a negative weight help invert the output? What happens if you change that weight to positive without changing the bias?

<details>
<summary><strong>Sample solution: NOT</strong></summary>

```python
weights_not = [-1.0]
bias_not = 0.5
```

The weighted sum is $z=-x+0.5$. For input 0, $z=0.5$ and output 1. For input 1, $z=-0.5$ and output 0. If the weight becomes positive while the bias remains $0.5$, both inputs produce output 1, so it is no longer NOT.

</details>

### Task 2D: Find another valid solution

Choose **OR or AND** and find a **different** pair of weights and bias that gives the same truth table. Test all four input pairs.

```python
# YOUR CODE HERE: new weights, new bias, and four test outputs.
```

**Explain:** Why can different parameter values give exactly the same classifications?

<details>
<summary><strong>Example answer</strong></summary>

For OR, `weights = [2.0, 2.0]` and `bias = -1.0` work. For AND, `weights = [2.0, 2.0]` and `bias = -3.0` work. These multiply all original weights and the bias by 2, so the sign of every weighted sum stays the same, preserving each output.

</details>

---

## 3. Visualize Linear Separability and XOR

One perceptron can create only **one straight decision boundary** in a two-feature space. Let's compare **AND**, which is linearly separable, with **XOR**, which is not.

### Task 3A: Predict before plotting

The inputs are the four corners of the unit square:

`(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`.

1. Sketch these four points on paper.
2. For AND, mark the output-0 points and output-1 points. Can you draw one straight separator?
3. Do the same for XOR. Can you draw **one** straight separator?
4. Predict which gate a single perceptron can represent perfectly.

### Task 3B: Plot AND and XOR

The plotting function below uses **blue for output 0** and **red for output 1**. Start by viewing the data **without** any decision boundary drawn for you.

```python
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs_and = np.array([0, 0, 0, 1])
outputs_xor = np.array([0, 1, 1, 0])


def plot_gate(labels, title, show_and_boundary=False):
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ["tab:blue" if label == 0 else "tab:red" for label in labels]
    ax.scatter(inputs[:, 0], inputs[:, 1], c=colors, s=180, zorder=3)

    for (x1, x2), label in zip(inputs, labels):
        ax.annotate(f"({x1},{x2}) → {label}", (x1, x2),
                    xytext=(5, 7), textcoords="offset points", fontsize=9)

    if show_and_boundary:
        x_line = np.linspace(-0.5, 1.5, 100)
        ax.plot(x_line, 1.5 - x_line, "k--", label="AND boundary: x₁ + x₂ = 1.5")
        ax.legend(loc="upper right")

    ax.set(title=title, xlabel="Input 1 (x₁)", ylabel="Input 2 (x₂)")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    plt.show()


plot_gate(outputs_and, "AND: Linearly separable")
plot_gate(outputs_xor, "XOR: Not linearly separable")
```

**Check your sketch:** Is the pattern what you expected? Where could you draw the line for AND? Why does no single line separate XOR perfectly?

<details>
<summary><strong>Code explanation: Plotting the points</strong></summary>

- `inputs[:, 0]` means all rows, first column (the first input feature).
- `inputs[:, 1]` means all rows, second column.
- `ax.scatter()` plots the four points and colors them by class.
- `ax.annotate()` labels each point with its coordinates and expected output.
- `ax.set_aspect("equal")` keeps equal distances along the two axes.

</details>

### Task 3C: Reveal an AND decision boundary

**After** you have drawn your own line, run:

```python
plot_gate(outputs_and, "AND with a decision boundary", show_and_boundary=True)
```

**Questions:**

1. Which points lie on the class-1 side of the line $x_1+x_2=1.5$?
2. How is this line related to the weights $[1,1]$ and bias $-1.5$ from Section 2?
3. Could moving or rotating a single line separate the XOR points? Explain.

<details>
<summary><strong>Sample explanation: The AND boundary</strong></summary>

The line $x_1+x_2=1.5$ comes directly from $z=x_1+x_2-1.5=0$. Only the point $(1,1)$ has $z\ge0$; the other three points have $z<0$. For XOR, the two class-1 points occupy diagonally opposite corners, so no single straight line can place both of them on one side while keeping both class-0 points on the other side.

</details>

---

## 4. The XOR Challenge

XOR produces output 1 if its inputs are different and output 0 when they are equal.

| Input 1 | Input 2 | Expected XOR |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

### Task 4A: Try and explain

Try **at least two different** sets of weights and bias in a single perceptron. For each attempt:

1. Record the weights and bias.
2. Compare all four predicted outputs with the XOR truth table.
3. Count how many of the four predictions are correct.
4. Explain why changing the single boundary cannot produce a perfect solution.

**You do not need to keep trying until it works: a single perceptron cannot represent XOR perfectly.** The goal is to discover and explain why.

```python
expected_xor = [0, 1, 1, 0]

# TRY TWO OR MORE PARAMETER SETS (examples below are only starting points)
trials = [
    ([1.0, 1.0], -0.5),
    ([1.0, 1.0], -1.5),
]

for trial_number, (weights, bias) in enumerate(trials, start=1):
    predicted = [perceptron(pair, weights, bias) for pair in gate_inputs]
    correct = sum(int(p == y) for p, y in zip(predicted, expected_xor))
    print(f"Trial {trial_number}: weights={weights}, bias={bias}")
    print(f"  Predictions: {predicted} | Expected: {expected_xor}")
    print(f"  Correct: {correct}/4")

# OPTIONAL: Replace the examples or add more trials of your own.
```

**Required reflection (write 3–5 sentences):**

- Why can one perceptron solve AND but not XOR?
- Use the **plot** and the idea of a **straight-line decision boundary** in your explanation.
- What might change if we connect several neurons in layers?

<details>
<summary><strong>Sample discussion answer</strong></summary>

A single perceptron predicts using the sign of one weighted sum, so its decision boundary is one straight line. AND is linearly separable: one line can isolate the point $(1,1)$ from the other three. XOR is not linearly separable: its two class-1 points lie at opposite corners, so no one line separates both classes correctly. Combining neurons into layers lets us combine multiple computations and build a nonlinear decision boundary.

</details>

### Optional mathematical challenge: Prove XOR is impossible

Suppose a perceptron returns 1 exactly when $w_1x_1+w_2x_2+b\ge0$. Write an inequality for each of the four XOR inputs, then show that the inequalities contradict each other.

<details>
<summary><strong>Optional proof</strong></summary>

To reproduce XOR, we would need:

1. $(0,0)\rightarrow0$: $b<0$.
2. $(1,0)\rightarrow1$: $w_1+b\ge0$.
3. $(0,1)\rightarrow1$: $w_2+b\ge0$.
4. $(1,1)\rightarrow0$: $w_1+w_2+b<0$.

Adding (2) and (3) gives $w_1+w_2+2b\ge0$. Since $b<0$, subtracting $b$ gives

$$w_1+w_2+b=(w_1+w_2+2b)-b>0.$$

That contradicts requirement (4), which says the same expression must be negative. Therefore, **no single perceptron can reproduce XOR**.

</details>

---

## 5. Takeaway: Why Do We Need More Than One Perceptron?

A single perceptron combines inputs into **one straight decision boundary**. This is enough for AND, OR, and NOT, but not for XOR. An MLP can use **hidden neurons and nonlinear activation functions** to combine computations in ways one perceptron cannot.

### Exit ticket

Before leaving, answer these questions **without looking at the sample solutions**:

1. What computation does a perceptron perform before applying its activation function?
2. What is the effect of changing a perceptron's bias while keeping the weights fixed?
3. Which point must an AND gate classify as 1?
4. Why does one straight decision boundary fail for XOR?
5. How might a hidden layer help?

<details>
<summary><strong>Exit-ticket answer guide</strong></summary>

1. A weighted sum (dot product) of the inputs, plus a bias.
2. It moves the decision boundary and can change which inputs reach the threshold.
3. Only $(1,1)$ for two binary inputs.
4. XOR's positive and negative classes occupy alternating corners of the square and cannot be separated by one straight line.
5. Hidden neurons can compute intermediate patterns that are combined to produce a more complex decision boundary.

</details>

**Next activity:** In **Activity 2**, you will build an MLP in Keras and attempt to solve XOR with a hidden layer. For now, focus on understanding the architecture and its behavior. We will study how training works internally later in the course.
