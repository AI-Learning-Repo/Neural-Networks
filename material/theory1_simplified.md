# Module 1 Theory — Simplified: From Characters to Attention

**A beginner-friendly guide to how Tiny-2.1 reads text**

> [!NOTE]
> **How to use this reading:** Read one short section, try its linked visualization, and answer its check-your-understanding question. Then complete the matching section of **Activity 1 — Simplified**. The expandable **More detail** boxes are optional.
>
> We use the same example throughout: `"2+2?\n4\n"`. In the displays below, `↵` means a newline (`\n`), not an extra character.

## What will you learn?

By the end of Module 1, you should be able to explain how Tiny-2.1:

1. changes characters into numerical IDs;
2. learns to predict the next character;
3. represents each character and its position as a vector;
4. uses attention to combine information from different positions;
5. uses Query, Key, Value, and a causal mask; and
6. performs attention with four parallel heads.

### Visualizations for this module

| Visualization | What it shows | Link |
|---|---|---|
| **1 — Generation and Training** | Predicting characters and training targets | [Open Visualization 1](https://ai-learning-repo.github.io/resources/demo/nlp/visualization_1_generation_and_training.html) |
| **2 — Token and Position Embeddings** | Character and position vectors | [Open Visualization 2](https://ai-learning-repo.github.io/resources/demo/nlp/visualization_2_token_and_position_embeddings.html) |
| **3 — Similarity Attention** | Similarity scores, softmax, context | [Open Visualization 3](https://ai-learning-repo.github.io/resources/demo/nlp/visualization_3_similarity_attention.html) |
| **4 — Q/K/V and Causal Attention** | Learned projections and looking only backward | [Open Visualization 4](https://ai-learning-repo.github.io/resources/demo/nlp/visualization_4_qkv_causal_attention.html) |
| **5 — Multi-Head Attention** | Scaling and four parallel heads | [Open Visualization 5](https://ai-learning-repo.github.io/resources/demo/nlp/visualization_5_multihead_attention.html) |

---

## Before we begin — Meet Tiny-2.1

Tiny-2.1 is a small **character-level language model**. Instead of generating a whole sentence in one step, it predicts **one character at a time**. That means it learns patterns such as:

```text
Question given to the model:  2+2?↵
Possible next character:     4
Next character after '4':     ↵  (stop)
```

This is an **illustration of the process**, not a guarantee of the trained model's exact answer.

Tiny-2.1 has a few settings that you will see in its code and in our activities:

| Code name | Value | Meaning in ordinary language |
|---|---:|---|
| `VOCAB` | `97` | Number of character IDs the model can represent |
| `CTX` | `128` | Maximum number of character positions handled at once |
| `D` | `96` | Number of values in each token's vector (the model width) |
| `L` | `4` | Number of Transformer blocks stacked one after another |
| `H` | `4` | Number of parallel attention heads **inside each block** |
| `HID` | `256` | Width of the MLP's **intermediate hidden layer**: `96 → 256 → 96` |

**Do not confuse `H` and `HID`:** `H=4` counts attention heads. `HID=256` counts the intermediate features used by the feed-forward network, which you will study in Module 2.

You may also see the letters `B` and `T` when reading tensor shapes:

- `B` = **batch size**: how many sequences are processed together.
- `T` = **sequence length**: how many character positions are in each sequence.
- `(B, T, D)` describes a collection of vectors. For one five-character prompt, the shape is `(1, 5, 96)`.

<details>
<summary>More detail — What is a tensor?</summary>

A **scalar** is one number, such as `5`. A **vector** is a list of numbers, such as `[1, 2, 3]`. A **matrix** is a grid of numbers. A **tensor** is a general name for these numerical structures, including higher-dimensional collections used in PyTorch.

A tensor of shape `(1, 5, 96)` contains one sequence, five positions, and 96 features at each position. We will use small 2D or 3D vectors on paper to understand the same operations.

</details>

---

## 1. Tokenization — Text becomes IDs

**Main idea:** A neural network needs numbers, so Tiny-2.1 gives each character its own integer ID.

For example:

```text
Text:       c    a    t
Character: 'c'  'a'  't'
ID:        69   67   86
```

The IDs are **labels**, not numerical measurements. The fact that `'t'` has a larger ID than `'a'` does not mean `'t'` is more important.

Tiny-2.1 also has two special IDs:

- `0` = padding, which can fill unused positions in a training batch;
- `1` = newline (`\n`), which separates the question from the answer and can end a generated answer.

**Worked example:**

```text
Text:   2   +   2   ?   ↵   4   ↵
ID:    20  13  20  33   1  22   1
```

The `2` appears twice, so its character ID is `20` both times. A character's **position** will be added later.

**See it:** [Visualization 1 — Generation and Training](https://ai-learning-repo.github.io/resources/demo/nlp/visualization_1_generation_and_training.html).

**Check your understanding:** Why do the two `2` characters have the same ID?

<details>
<summary>Answer</summary>

Because they are the same character. Tokenization identifies the character, not where it appears in the sequence.

</details>

<details>
<summary>More detail — How are the IDs calculated?</summary>

For printable ASCII characters, Tiny-2.1 uses the simple rule:

$$
\mathrm{ID}(c) = \mathrm{ASCII}(c) - 30
$$

Newline is handled separately as ID `1`, and padding is ID `0`. For example, `ASCII('2') = 50`, so its ID is $50 - 30 = 20$.

</details>

---

## 2. Learning to predict the next character

**Main idea:** During training, the model sees some characters and learns to predict what comes next.

Consider a short word:

```text
Full text:   c   a   t   ↵
Input:       c   a   t
Target:      a   t   ↵
```

The targets are simply shifted one position to the left. After `c`, the correct next character is `a`. After `ca`, it is `t`.

### Why do we use a loss mask?

Tiny-2.1 trains on question-and-answer examples. It should learn to produce the **answer**, not receive a penalty for failing to guess the user's question.

For our familiar example:

```text
Full text:     2   +   2   ?   ↵   4   ↵
Input:         2   +   2   ?   ↵   4
Next target:   +   2   ?   ↵   4   ↵
Loss mask:     0   0   0   0   1   1
```

- A `0` means **ignore this target when calculating training loss**.
- A `1` means **this target matters for training**.

Only the predictions for `4` and the final newline count toward the training loss in this example.

**Important distinction:** The **loss mask** is used when learning from answers. It is different from the **causal attention mask** in Section 6, which prevents looking ahead inside the Transformer.

**See it:** [Visualization 1 — Generation and Training](https://ai-learning-repo.github.io/resources/demo/nlp/visualization_1_generation_and_training.html).

**Check your understanding:** Why does the final newline have a loss-mask value of `1`?

<details>
<summary>Answer</summary>

Because the model must learn where the answer ends. Predicting the final newline teaches it when to stop generating.

</details>

---

## 3. Embeddings — Character vectors and position vectors

**Main idea:** A character ID is just a label. To do useful calculations, the model looks up a **vector**: a list of learned numbers.

Tiny-2.1 uses **two** lookup tables:

- **Token embedding:** Which character is this?
- **Position embedding:** Where is this character in the sequence?

It **adds** their vectors together.

### Worked example: the two `2` characters in `2+2?`

For easy arithmetic, suppose vectors have **three values** rather than Tiny-2.1's real width of 96.

The token embedding for `2` is the same at both places:

$$
E[\text{'2'}] = [1,\;0,\;0.2]
$$

But the two positions have different position embeddings:

$$
P[0] = [0.1,\;0.2,\;0], \qquad P[2] = [0.4,\;-0.1,\;0.3]
$$

So the combined vectors are:

$$
\begin{aligned}
X_0 &= [1,\;0,\;0.2] + [0.1,\;0.2,\;0] = [1.1,\;0.2,\;0.2] \\
X_2 &= [1,\;0,\;0.2] + [0.4,\;-0.1,\;0.3] = [1.4,\;-0.1,\;0.5]
\end{aligned}
$$

The two characters are identical, but the **combined vectors differ** because their positions differ.

For the real model, the calculation is the same, but each vector contains **96 numbers**, not three.

**Code names:** `model.emb` is the learned token-embedding table and `model.pos` is the learned position-embedding table. The model adds their outputs.

**See it:** [Visualization 2 — Token and Position Embeddings](https://ai-learning-repo.github.io/resources/demo/nlp/visualization_2_token_and_position_embeddings.html).

**Check your understanding:** Why does the second `2` get a different final vector from the first `2`?

<details>
<summary>Answer</summary>

They use the same token embedding but different position embeddings, so the sums are different.

</details>

<details>
<summary>More detail — One formula</summary>

For the token at position $t$:

$$
X_t = E[\mathrm{ID}_t] + P[t]
$$

`E` and `P` are learned during training; they are not handwritten rules.

</details>

---

## 4. Attention — Gathering information from other positions

**Main idea:** A character often needs information from other characters to predict what comes next.

For example, the question mark in `2+2?` may need information about the two digits and the plus sign. **Attention** provides a way for one position to gather information from other positions.

To learn the idea, start with a **simple similarity-based example**. This is a *teaching model*, **not an extra layer in the real Tiny-2.1 architecture**.

### Worked example with two vectors

Suppose we have:

$$
x_0 = [1,\;0], \qquad x_1 = [0,\;1]
$$

The first vector's similarity scores, calculated using dot products, are:

$$
\begin{aligned}
x_0 \cdot x_0 &= 1 \\
x_0 \cdot x_1 &= 0
\end{aligned}
$$

**Softmax** changes the scores $[1,0]$ into attention weights of approximately $[0.73,0.27]$. The weights are positive and add up to one.

The new context vector is a **weighted combination**:

$$
0.73[1,\;0] + 0.27[0,\;1] = [0.73,\;0.27]
$$

That is the key idea: **use scores to decide how much information to gather from each position**.

In this first teaching example, we have *not yet added* Query/Key/Value projections or causal masking. The real model uses both.

**See it:** [Visualization 3 — Similarity Attention](https://ai-learning-repo.github.io/resources/demo/nlp/visualization_3_similarity_attention.html).

**Check your understanding:** What do the weights `0.73` and `0.27` tell us?

<details>
<summary>Answer</summary>

They tell us how strongly the first position blends information from the two vectors in this example: 73% from the first and 27% from the second.

</details>

<details>
<summary>More detail — Matrix version of simple attention</summary>

With input matrix $X$, the simple, unprojected teaching version is:

$$
\mathrm{Context}(X) = \mathrm{softmax}(XX^T)X
$$

- $XX^T$ calculates all pairwise dot-product scores.
- `softmax` converts each row of scores into weights that add up to one.
- Multiplication by $X$ creates the weighted context vectors.

The real Tiny-2.1 attention instead uses learned Query, Key, and Value vectors, scaling, and a causal mask.

</details>

---

## 5. Query, Key, Value — More flexible attention

**Main idea:** A token does not always need to pay attention to tokens that are numerically similar to itself. It may need information from a *different kind* of token.

For example, `?` may need to attend to `2`, even though they are different characters.

Imagine searching for a video:

| Attention name | Search analogy | Main question |
|---|---|---|
| **Query (`Q`)** | What you type into the search box | **What am I looking for?** |
| **Key (`K`)** | The tags describing each video | **What information might I match?** |
| **Value (`V`)** | The content of the video | **What information can I give?** |

Tiny-2.1 takes the same input representation and creates **three different versions** of it using learned linear layers.

```text
          Input vector X
             / | \
            /  |  \
           v   v   v
           Q   K   V
           |   |   |
        match scores  information to combine
```

A Query from the question mark can match a Key from a digit, and attention then collects information from the corresponding Value.

### What do the code names mean?

- `blk.q` computes Query vectors.
- `blk.k` computes Key vectors.
- `blk.v` computes Value vectors.
- `blk.o` is the **output projection** that combines the attention-head results. It is not a fourth type of Q/K/V.
- `blk` is a short variable name for one Transformer **block**.

**See it:** [Visualization 4 — Query, Key, Value, and Causal Attention](https://ai-learning-repo.github.io/resources/demo/nlp/visualization_4_qkv_causal_attention.html).

**Check your understanding:** If `Q` is what a position is seeking, what does `V` represent?

<details>
<summary>Answer</summary>

`V` is the information that gets combined into the output when the Query and Key match strongly.

</details>

<details>
<summary>More detail — How the model computes Q, K, and V</summary>

$$
Q = XW_Q^T, \qquad K = XW_K^T, \qquad V = XW_V^T
$$

$W_Q$, $W_K$, and $W_V$ are learned weight matrices. In PyTorch, `nn.Linear` stores its weight matrix as `(out_features, in_features)`, which is why the matrix appears **transposed** in these equations.

The projection layers are linear; softmax later makes the attention weights depend nonlinearly on their scores.

</details>

---

## 6. Causal masking — No looking into the future

**Main idea:** A model generating text must not use characters that have not yet been generated.

Consider `2+2?`. The causal attention mask looks like this:

```text
Can this position use information from this column?
                2   +   2   ?
First  2       [1   0   0   0]
       +       [1   1   0   0]
Second 2       [1   1   1   0]
       ?       [1   1   1   1]

1 = allowed        0 = blocked
```

The first `2` cannot see the `?` because the question mark is **in the future** relative to that position. The `?`, however, can use all the preceding characters.

Why is this especially important in training? The model may receive the entire sequence `2+2?↵4↵` at once for efficiency. Without the causal mask, the position just before `4` could read the `4` from the future instead of predicting it.

This mask operates **inside attention**, both during training and when processing a prompt for generation. It is **not** the loss mask from Section 2.

**See it:** [Visualization 4 — Causal Masking](https://ai-learning-repo.github.io/resources/demo/nlp/visualization_4_qkv_causal_attention.html).

**Check your understanding:** Can the second `2` attend to the final `?` in `2+2?`?

<details>
<summary>Answer</summary>

No. The question mark comes later in the sequence. Causal attention can use the current position and positions before it, but not future positions.

</details>

<details>
<summary>More detail — Why masked scores become zero probability</summary>

Before softmax, the model replaces disallowed scores by $-\infty$. Since $e^{-\infty}=0$, those positions receive attention weight zero.

The full operation is:

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
$$

where $M_{ij}=0$ for allowed positions $j \le i$, and $M_{ij}=-\infty$ for future positions $j>i$. We explain $d_k$ next.

</details>

---

## 7. Scaling and multi-head attention — Four views of the sequence

**Main idea:** Tiny-2.1 uses **four attention heads in parallel** inside each Transformer block.

The model's vector width is `D = 96`, so its four heads each work with:

$$
d_k = \frac{D}{H} = \frac{96}{4} = 24
$$

Each head calculates its own attention weights and context vectors. The head outputs are then **concatenated** (placed side-by-side), returning to 96 features. The learned output layer `o` combines those features.

```text
Input representation: 96 features
          |
  Q, K, V projections
          |
  +-------+-------+-------+-------+
  |       |       |       |       |
Head 0  Head 1  Head 2  Head 3
 24-d    24-d    24-d    24-d
  |       |       |       |       |
  +-------+-------+-------+-------+
          |
  Concatenate: 96 features
          |
  Output projection 'o': 96 features
```

The four heads **can** learn to use information differently, but a heatmap alone does not prove that one particular head performs a particular reasoning task.

### Why divide attention scores by a square root?

Longer vectors can produce larger dot products. Very large differences in scores make softmax too sharply concentrated. So scaled dot-product attention divides the Query–Key scores by $\sqrt{d_k}$.

For our small 3D visualizations, the factor is $\sqrt{3}$. In the real Tiny-2.1 heads, the factor is $\sqrt{24}$.

**See it:** [Visualization 5 — Scaling and Multi-Head Attention](https://ai-learning-repo.github.io/resources/demo/nlp/visualization_5_multihead_attention.html).

**Check your understanding:** Does Tiny-2.1 have four blocks **or** four heads?

<details>
<summary>Answer</summary>

Both! It has **four blocks in sequence**, and **each block has four attention heads in parallel**. We'll assemble and stack the complete blocks in Module 2.

</details>

<details>
<summary>More detail — What happens to the dimensions?</summary>

At one stage, Q, K, and V have shape `(B, T, 96)`. Splitting a projection into four heads changes its logical shape to `(B, 4, T, 24)`.

The context outputs of four heads are concatenated into `(B, T, 96)`. The output projection `o` then maps 96 features back to 96. The model does **not** sum the four head vectors into a 24-dimensional vector.

</details>

---

## Module 1 — What should you remember?

1. **Characters become IDs**, and IDs are just labels.
2. **Embeddings** convert IDs and positions into useful vectors.
3. **Attention** blends information using learned scores and weights.
4. **Q, K, and V** make attention flexible.
5. **Causal masking** prevents information from future positions from leaking backward.
6. **Four heads** compute attention in parallel inside each block.

**Next:** In Module 2, we'll see where attention fits inside a full Transformer block, and how the complete model chooses and generates its next character.

**Practice:** Continue with *Activity 1 — Simplified*, Part B.

### Optional further reading

- [Tiny-2.1 project and source code](https://github.com/WollyMonjac/Tiny-2.1)
- [Build a Large Language Model From Scratch — Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch)
- [The original Transformer paper: Attention Is All You Need](https://arxiv.org/abs/1706.03762)

