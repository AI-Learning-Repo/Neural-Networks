# From Small Neural Networks to Large-Scale Training

*An introductory connection between multilayer perceptrons, deep neural networks, automatic differentiation, and stochastic gradient descent*

## Purpose of this reading

The earlier examples used one or two weights to make predictions, calculate a loss, and update the model. Larger neural networks follow the **same fundamental learning cycle**:

**Forward pass → Loss → Backpropagation → Parameter update → Repeat**

This reading introduces three questions without requiring new calculus techniques:

1. How does a small neural network become a deep neural network?
2. How can a network with millions or billions of parameters calculate gradients?
3. How can training proceed when the dataset contains millions of observations?

The central distinction is that **network architecture**, **gradient calculation**, and **selection of training observations** answer different questions.

---

## 1. From a neuron to an MLP and a deep neural network

### 1.1 Starting with a familiar neuron

A two-input linear neuron calculates a weighted sum:

$$
z=w_1x_1+w_2x_2+b
$$

For regression, the output can simply be this weighted sum. This uses the **identity (linear) activation**: the value passes through without changing. For binary classification, a sigmoid activation instead converts the weighted sum into a probability. A single sigmoid output neuron with no hidden layer is logistic regression, not an MLP with hidden layers.

Adding **hidden layers** allows the network to carry out several stages of computation before producing its output. Hidden layers commonly apply nonlinear activation functions, which allow the network to represent more complex relationships.

### 1.2 What is a multilayer perceptron (MLP)?

A **multilayer perceptron (MLP)** is a type of *feedforward neural network*. It typically contains an input layer, one or more hidden layers of fully connected neurons, and an output layer. Information moves from the inputs, through the hidden layers, to the output.

**Example — an MLP with one hidden layer:**

```mermaid
graph LR
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef hidden fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;

    subgraph Input_Layer["Input Layer"]
        x1(("x₁")):::input
        x2(("x₂")):::input
    end

    subgraph Hidden_Layer["Hidden Layer"]
        h1(("h₁")):::hidden
        h2(("h₂")):::hidden
    end

    subgraph Output_Layer["Output Layer"]
        out(("ŷ")):::output
    end

    x1 --> h1
    x1 --> h2
    x2 --> h1
    x2 --> h2
    h1 --> out
    h2 --> out
```

Each connection has a trainable weight, and neurons commonly have trainable biases. The diagram omits their labels to keep the architecture easy to read.

### 1.3 When is an MLP also a DNN?

A **deep neural network (DNN)** has multiple hidden or processing layers. The word *deep* describes the network's depth; *MLP* describes a particular type of network architecture.

An MLP may be relatively shallow or deep. A **deep MLP is also a DNN**, but not every DNN is an MLP. Other deep architectures include convolutional neural networks and Transformers.

**Example — a deeper MLP:**

```mermaid
graph LR
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef hidden fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;

    subgraph Input_Layer["Input Layer"]
        x1(("x₁")):::input
        x2(("x₂")):::input
    end

    subgraph Hidden_Layer_1["Hidden Layer 1"]
        h11(("h₁")):::hidden
        h12(("h₂")):::hidden
    end

    subgraph Hidden_Layer_2["Hidden Layer 2"]
        h21(("h₃")):::hidden
        h22(("h₄")):::hidden
    end

    subgraph Output_Layer["Output Layer"]
        out(("ŷ")):::output
    end

    x1 --> h11
    x1 --> h12
    x2 --> h11
    x2 --> h12
    h11 --> h21
    h11 --> h22
    h12 --> h21
    h12 --> h22
    h21 --> out
    h22 --> out
```

There is no universal minimum number of layers that defines *deep*. These diagrams show the difference in structure, not a fixed depth threshold.

**Main idea:** A more complex architecture introduces more layers, operations, and trainable parameters. It does **not** replace the basic learning cycle.

---

## 2. From three parameters to billions: how are gradients calculated?

### 2.1 The same derivative principle still applies

For a neuron with two weights and a bias, the trainable parameters are $w_1$, $w_2$, and $b$. Backpropagation calculates three partial derivatives:

$$
\frac{\partial L}{\partial w_1},\qquad
\frac{\partial L}{\partial w_2},\qquad
\frac{\partial L}{\partial b}
$$

Every independently trainable scalar parameter has a corresponding loss sensitivity. If a hypothetical network has billions of *trainable* parameters participating in the loss, its gradient calculation can involve billions of components.

The principle does not change: **the chain rule connects the parameters to the final loss**. What changes is the number of operations and derivatives that must be handled.

### 2.2 Why calculate gradients with software?

Calculating a few derivatives by hand is useful for understanding backpropagation. Repeating a separate manual calculation for every weight in a large network is not practical.

Libraries such as **PyTorch** and **TensorFlow** use **automatic differentiation**. At a high level, the process is:

1. The forward pass computes predictions and the loss.
2. The differentiation system records or traces how the calculations depend on one another.
3. Starting from the loss, it applies the chain rule backward through those operations.
4. The resulting gradients become available to an **optimizer**, which performs the parameter updates.

This is commonly called *reverse-mode automatic differentiation*. It reuses intermediate calculations rather than rerunning the entire model independently for every parameter. The computation still requires resources and grows with model complexity.

```mermaid
flowchart LR
    classDef compute fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;

    A["Inputs and current parameters"] --> B["Forward pass"]:::compute
    B --> C["Prediction and loss"]:::output
    C --> D["Automatic differentiation: calculate gradients"]:::compute
    D --> E["Optimizer: update parameters"]:::output
    E -->|"Next step"| B
```

### 2.3 Backpropagation is not the parameter update

| Task | PyTorch example | TensorFlow example |
|---|---|---|
| Calculate gradients | `loss.backward()` | `GradientTape` and `tape.gradient(...)` |
| Update parameters | `optimizer.step()` | `optimizer.apply_gradients(...)` |

The names differ, but the distinction is the same as in the hand calculations:

**Backpropagation calculates how loss depends on the parameters. An optimizer uses those gradients to change the parameters.**

The library does not change the mathematical purpose of a gradient; it makes gradient calculation practical at a much larger scale.

---

## 3. From gradient descent to SGD: how much data is used for each update?

### 3.1 A second scaling challenge: the number of observations

Large networks may have many parameters. Training datasets may also contain very many **observations**. These are two different kinds of scale.

In **full-batch gradient descent**, each update uses the average gradient calculated from the **entire training dataset**. If the dataset contains one million observations, a full-batch update requires processing those one million observations first.

A different approach is to update the parameters after processing a smaller number of observations.

### 3.2 Three ways to choose the data for an update

| Method | Observations contributing to one update | Main consequence |
|---|---|---|
| **Full-batch gradient descent** | Entire dataset | One full-data gradient per update |
| **Mini-batch gradient descent** | A small group of observations | More frequent updates using approximate full-data gradients |
| **Pure stochastic gradient descent (SGD)** | One observation | Very frequent updates; individual gradients can vary substantially |

In deep-learning discussions, the term **SGD** is also commonly used for an SGD optimizer trained with *mini-batches*. The batch size makes the precise meaning clear.

**Example:** With **1,000,000 observations**, processed once per epoch:

- Full-batch gradient descent makes **1 update per epoch**.
- Mini-batches containing **100 observations** produce **10,000 updates per epoch**.
- One-observation updates produce **1,000,000 updates per epoch**.

An **epoch** is one complete pass through the dataset. An **iteration** or **step** is one parameter update. More updates do not automatically mean faster or better training: gradient variability and computational efficiency also matter.

### 3.3 What stays the same?

All three methods use the familiar gradient-descent idea:

$$
\theta_{\mathrm{new}}=\theta_{\mathrm{old}}-\eta g
$$

Here, $\theta$ represents a trainable parameter (or the parameter collection), $\eta$ is the learning rate, and $g$ is the gradient calculated from the selected observations.

What changes is **which observations contribute to the calculated gradient**. Because different observations may produce different gradients, single-example and mini-batch updates may move in different directions from a full-batch update.

A small numerical illustration uses the same current weight, $w=10$, and learning rate, $0.1$. Suppose the gradients from four observations are $4$, $2$, $-2$, and $0$:

| Method | Gradient selected or averaged | Updated weight |
|---|---:|---:|
| Full batch: all four observations | $1$ | $9.9$ |
| Mini-batch: first two observations | $3$ | $9.7$ |
| Pure SGD: third observation | $-2$ | $10.2$ |

The update rule has **not changed**. The gradient differs because the selected data differ. These are illustrative single updates; in actual training, gradients are calculated again after each parameter update.

---

## 4. Connecting all three ideas

The earlier one-weight and two-weight examples already contain the essential mathematics for understanding much larger systems:

| Question | Main concept | What it determines |
|---|---|---|
| **What computation does the model perform?** | MLP or another neural-network architecture | How inputs become predictions |
| **How is the loss sensitivity calculated?** | Backpropagation / automatic differentiation | Gradients for trainable parameters |
| **Which data contribute to this update?** | Full batch, mini-batch, or single-example SGD | The gradient used at that step |
| **How are parameters changed?** | Gradient-descent optimizer and learning rate | The next parameter values |

The same sequence remains recognizable in small and large models:

**Forward pass → Loss → Calculate gradients → Update parameters → Repeat**

A deep network adds computational layers. Automatic differentiation makes the chain-rule calculations manageable. Mini-batch SGD changes how much training data contributes to each update. These ideas work together, but they do **different jobs**.

## Quick concept check

**Question 1. Is every MLP a deep neural network?**

<details>
<summary>Show answer</summary>

No. An MLP can be shallow or deep. A sufficiently deep MLP is a DNN, while many DNNs use architectures other than MLPs.

</details>

**Question 2. Does backpropagation update the weights?**

<details>
<summary>Show answer</summary>

No. Backpropagation calculates gradients by applying the chain rule. A separate optimizer uses these gradients to update trainable parameters.

</details>

**Question 3. What changes when full-batch gradient descent is replaced with mini-batch SGD?**

<details>
<summary>Show answer</summary>

The selected training observations used for each gradient calculation change. The basic chain rule and gradient-descent update idea remain the same, but updates occur more frequently and gradients can vary between mini-batches.

</details>

---

**Key takeaway:** Larger neural networks still learn through the same sequence practiced with a few weights. **Architecture** specifies the computations, **backpropagation** calculates gradients, and **the optimizer** updates parameters using gradients from the selected training observations.
