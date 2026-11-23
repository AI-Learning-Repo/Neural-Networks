# Activity 2 — Simplified

This worksheet is a beginner-friendly introduction to the **complete Transformer block** in Tiny-2.1.

It has two parts:

- **Part A:** run Tiny-2.1 and observe what it does
- **Part B:** understand the main ideas behind RMSNorm, residual connections, the MLP, the full Transformer block, and autoregressive generation

This simplified worksheet is designed for beginners. Most tasks follow the pattern:

1. **Run** the code
2. **Observe** the result
3. **Change** one small thing
4. **Explain** what happened

You do **not** need to implement Tiny-2.1 from scratch.

---

## Part A — Run and Explore Tiny-2.1

### A1. Set up the repository

Run this cell first.

```python
!git clone https://github.com/WollyMonjac/Tiny-2.1.git
%cd /content/Tiny-2.1/training
```

### A2. Load the pretrained model

```python
import train2

model = train2.load_model("tiny2.pt")
model.eval()

print("Tiny-2.1 loaded successfully!")
```

**Expected output**

```text
Tiny-2.1 loaded successfully!
```

### A3. Ask Tiny-2.1 a few questions

```python
questions = [
    "what is 2+2?",
    "what is the capital of france?",
    "what is the largest planet?",
    "what is the capital of mars?",
]

answers = train2.answer_many(model, questions)

for q, a in answers.items():
    print(f"Q: {q}")
    print(f"A: {a}\n")
```

### A4. Observe and discuss

1. Which answers seem correct?
2. Which answer seems incorrect or made up?
3. Why might a small model fail on some questions?

<details>
<summary>Hint</summary>
Think about the size of the model, the amount of training data, and whether the model really "knows" facts.
</details>

<details>
<summary>Possible answer</summary>
Some answers may be correct, but a small model can still make up plausible-sounding answers. This is sometimes called hallucination.

Tiny-2.1 has limited capacity and limited training data compared with modern large language models.
</details>

### A5. Inspect the architecture

```python
print(model)
print("Total parameters:", sum(p.numel() for p in model.parameters()))
```

### A6. Quick questions

1. How many Transformer blocks does Tiny-2.1 use?
2. What is the hidden width of the model?
3. How many attention heads does each block use?

<details>
<summary>Hint</summary>
Look for the repeated block structure and the configuration values discussed in class.
</details>

<details>
<summary>Answer</summary>
Tiny-2.1 uses:
- 4 Transformer blocks
- hidden width 96
- 4 attention heads
</details>

---

# Part B — Understand the Complete Transformer

In this part, you will study the main building blocks that were added after attention:

- RMSNorm
- residual connections
- the feed-forward network (MLP)
- the complete Transformer block
- autoregressive generation

We will mostly use **small worked examples** rather than full 96-dimensional calculations.

---

## B1 — RMSNorm

RMSNorm helps keep vector magnitudes more stable.

The formula is:

\[
\operatorname{RMSNorm}(x)=
\frac{x}{\sqrt{\frac{1}{D}\sum_i x_i^2 + \epsilon}}
\odot g
\]

Here:
- \(x\) is the input vector
- \(D\) is the number of dimensions
- \(\epsilon\) is a small constant
- \(g\) is a learned gain vector

### B1.1 A small example

Compare these two vectors:

- `x1 = [1, 7, 1, 7]`
- `x2 = [10, 70, 10, 70]`

Run the code:

```python
import math
import torch

x1 = torch.tensor([1.0, 7.0, 1.0, 7.0])
x2 = torch.tensor([10.0, 70.0, 10.0, 70.0])
g  = torch.tensor([1.0, 2.0, 1.0, 0.5])
eps = 1e-5

def rmsnorm(x, g, eps=1e-5):
    rms = torch.sqrt(torch.mean(x * x) + eps)
    return (x / rms) * g

print("RMSNorm(x1) =", rmsnorm(x1, g, eps))
print("RMSNorm(x2) =", rmsnorm(x2, g, eps))
```

### B1.2 Observe

1. Are the two outputs similar or very different?
2. Why do we say RMSNorm helps make the scale more stable?

<details>
<summary>Hint</summary>
The second vector is 10× larger, but the pattern is the same.
</details>

<details>
<summary>Answer</summary>
The outputs should be very similar. RMSNorm divides by the root-mean-square value, so scaling the whole vector up also scales up the denominator.

This means the normalized output depends more on the pattern than on the raw size.
</details>

### B1.3 Change one thing

Change the gain vector `g` to:

```python
torch.tensor([1.0, 1.0, 1.0, 1.0])
```

Run again.

**Question:** What role does the gain vector play?

<details>
<summary>Answer</summary>
The gain vector allows the model to re-scale each feature after normalization. RMSNorm stabilizes the input, and the gain lets the model still learn useful per-feature scaling.
</details>

---

## B2 — Residual Connections

A residual connection adds the original input back to the update produced by a sublayer.

\[
\text{output} = \text{input} + \text{update}
\]

### B2.1 A very simple example

```python
input_value = 5.0
update1 = 2.0
update2 = 0.0

print("With update 2:", input_value + update1)
print("With update 0:", input_value + update2)
```

### B2.2 Observe

1. What happens when the update is zero?
2. Why is this useful?

<details>
<summary>Hint</summary>
If the update is zero, the original information is still preserved.
</details>

<details>
<summary>Answer</summary>
If the update is zero, the output is still the original input.

This gives the network a shortcut path. It helps preserve information and also helps training by making it easier for signals and gradients to move through many layers.
</details>

### B2.3 Connect to Tiny-2.1

Each Transformer block in Tiny-2.1 has **two** residual additions:

1. after attention
2. after the MLP

**Question:** Why might a block need two residual connections instead of one?

<details>
<summary>Possible answer</summary>
Because both the attention sublayer and the MLP produce updates. Each update is added to the running representation instead of replacing it.
</details>

---

## B3 — Feed-Forward Network (MLP)

After attention, each token representation is processed by a small neural network:

\[
\text{MLP}(h)=W_2\,\mathrm{ReLU}(W_1h)
\]

The important idea is:

- **attention** mixes information across positions
- **the MLP** processes each position independently

### B3.1 ReLU first

Run this cell:

```python
x = torch.tensor([2.0, -3.0, 0.5, -1.0])
print(torch.relu(x))
```

**Expected output**

```text
tensor([2.0000, 0.0000, 0.5000, 0.0000])
```

### B3.2 Observe

1. Which values changed?
2. What does ReLU do?

<details>
<summary>Answer</summary>
ReLU keeps positive values unchanged and turns negative values into zero.
</details>

### B3.3 A tiny 2 → 4 → 2 example

Run this code:

```python
h = torch.tensor([1.0, -1.0])

W1 = torch.tensor([
    [ 1.0,  0.0],
    [ 0.0,  1.0],
    [ 1.0,  1.0],
    [-1.0,  1.0],
])

W2 = torch.tensor([
    [0.8, 0.0, 0.0, 0.0],
    [0.0, 0.4, 0.0, 0.0],
])

hidden_pre = W1 @ h
hidden_post = torch.relu(hidden_pre)
out = W2 @ hidden_post

print("Before ReLU:", hidden_pre)
print("After ReLU:", hidden_post)
print("Output:", out)
```

### B3.4 Change one thing

Change the input vector from:

```python
[1.0, -1.0]
```

to:

```python
[1.0, 1.0]
```

Run again.

**Question:** Why did the output change?

<details>
<summary>Possible answer</summary>
The hidden neurons receive different weighted sums, so the ReLU outputs also change. That changes the final output.
</details>

---

## B4 — The Complete Transformer Block

Now let's connect the pieces.

One Transformer block in Tiny-2.1 works like this:

1. RMSNorm
2. multi-head causal attention
3. residual addition
4. RMSNorm
5. MLP (`f1 → ReLU → f2`)
6. residual addition

### B4.1 Track the shapes

Read the table:

| Operation | Shape |
|---|---|
| Input embeddings | `(1, 5, 96)` |
| RMSNorm | `(1, 5, 96)` |
| Attention output | `(1, 5, 96)` |
| First residual | `(1, 5, 96)` |
| `f1` | `(1, 5, 256)` |
| ReLU | `(1, 5, 256)` |
| `f2` | `(1, 5, 96)` |
| Second residual | `(1, 5, 96)` |

### B4.2 Observe

1. Which operation expands the hidden size?
2. Why must the MLP return to 96 dimensions before the residual addition?

<details>
<summary>Hint</summary>
Residual addition requires both tensors to have the same shape.
</details>

<details>
<summary>Answer</summary>
The `f1` layer expands the representation from 96 to 256 dimensions.

The `f2` layer projects it back to 96 dimensions so it can be added to the running representation through the residual connection.
</details>

### B4.3 Weight tying idea

Tiny-2.1 uses the same embedding table at the input and output.

Input side:

- character ID → embedding vector

Output side:

- final hidden vector → dot product with all embedding rows → character scores

Run this toy example:

```python
h = torch.tensor([0.8, 0.6, 0.2])
E = torch.tensor([
    [1.0, 0.0, 0.0],  # '2'
    [0.0, 1.0, 0.0],  # '+'
    [0.0, 0.0, 1.0],  # '?'
    [1.0, 1.0, 0.0],  # '4'
    [0.2, 0.2, 0.9],  # '\n'
])

logits = h @ E.T
print(logits)
print("Best index:", int(torch.argmax(logits)))
```

### B4.4 Explain

What does the largest logit mean?

<details>
<summary>Answer</summary>
The largest logit corresponds to the character the model currently prefers as the next character.
</details>

---

## B5 — Autoregressive Generation

Tiny-2.1 generates text one character at a time.

The loop is:

1. run the model on the current prompt
2. take the logits at the last position
3. choose the next character
4. append it to the prompt
5. repeat until newline

### B5.1 A tiny demonstration

Run this code:

```python
prompt = "2+2?\n"
ids = train2.encode(prompt)
print("Initial prompt IDs:", ids)
print("Decoded back:", train2.decode(ids))
```

### B5.2 Generate with Tiny-2.1

```python
x = torch.tensor([train2.encode("2+2?\n")])

generated = []

with torch.no_grad():
    for step in range(20):
        logits = model(x)[:, -1, :]
        next_id = int(logits.argmax(dim=-1).item())

        if next_id <= 1:
            break

        generated.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]])], dim=1)

print("Generated IDs:", generated)
print("Generated text:", train2.decode(generated))
```

### B5.3 Observe

1. What text was generated?
2. Why does the loop stop?
3. What happens if you change the starting prompt?

Try changing the prompt from `"2+2?\n"` to:

- `"what is the capital of france?\n"`
- `"what is the largest planet?\n"`

### B5.4 Explain

Why is this called **autoregressive** generation?

<details>
<summary>Hint</summary>
Every new character becomes part of the input for the next step.
</details>

<details>
<summary>Answer</summary>
It is called autoregressive because the model generates one character at a time, and each predicted character is appended to the prompt and used to predict the next one.
</details>

---

## B6 — Optional Background: Training and Model Size

This final section is **reading only**.

You do not need to run any code here.

### Training loss

During training, Tiny-2.1 compares its predicted next-character scores with the correct next characters.

A loss function measures how wrong the predictions are. Training adjusts the model's weights to reduce that loss.

### Model size

Tiny-2.1 is small enough to run in very limited environments.

For classroom work, we use the normal PyTorch checkpoint `tiny2.pt`. A smaller deployment format also exists for DOS, but you do not need it for this activity.

<details>
<summary>Optional technical note: how the DOS model file can be loaded into PyTorch</summary>

The project also contains a compact binary model file used by the DOS implementation. The function below reads its format and reconstructs a compatible PyTorch model. **This is reference code only. You do not need to run or understand it for this activity.**

```python
# Optional reference only — not needed for this activity
import struct
import numpy as np
import torch
import train2

def load_v2_bin(path):
    """Read a Tiny-2.1 TNY2 weight file into a PyTorch model."""
    with open(path, "rb") as f:
        b = f.read()

    assert b[:4] == b"TNY2", "Expected a TNY2 weight file"
    D, L, H, HID = struct.unpack_from("<4h", b, 4)
    off = [12]

    def table(rows, cols):
        w = np.empty((rows, cols), dtype=np.float32)
        for i in range(rows):
            scale = struct.unpack_from("<f", b, off[0])[0]
            w[i] = np.frombuffer(
                b, np.int8, cols, off[0] + 4
            ).astype(np.float32) * scale
            off[0] += 4 + cols
        return torch.tensor(w)

    def gains():
        g = np.frombuffer(b, "<f4", D, off[0]).copy()
        off[0] += 4 * D
        return torch.tensor(g)

    m = train2.Model(train2.Cfg(D, L, H, HID))
    with torch.no_grad():
        m.emb.weight.copy_(table(train2.VOCAB, D))
        m.pos.weight.copy_(table(train2.CTX, D))
        for blk in m.blocks:
            blk.n1.g.copy_(gains())
            for linear in (blk.q, blk.k, blk.v, blk.o):
                linear.weight.copy_(table(D, D))
            blk.n2.g.copy_(gains())
            blk.f1.weight.copy_(table(HID, D))
            blk.f2.weight.copy_(table(D, HID))
        m.nf.g.copy_(gains())

    assert off[0] == len(b), "Binary data length did not match model dimensions"
    m.eval()
    return m
```

</details>

---

# Summary

In this simplified activity, you explored:

- how RMSNorm stabilizes vector magnitudes
- how residual connections preserve information
- how the MLP processes each token representation
- how a complete Transformer block fits together
- how Tiny-2.1 generates text one character at a time

If you want a greater challenge, you can use the **detailed** version of Activity 2 afterward.
