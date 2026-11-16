# Part 1 Visualizations: From Pixels to Learned Features


## Setup: Load Fashion-MNIST and prepare one test image

If an import is missing in Colab, run `%pip install numpy matplotlib tensorflow pillow requests scipy` in a notebook cell first.

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import urllib.request

keras.utils.set_random_seed(42)

(x_train_raw, y_train), (x_test_raw, y_test) = keras.datasets.fashion_mnist.load_data()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

x_train = x_train_raw.astype("float32") / 255.0
x_test = x_test_raw.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]  # shape: (60000, 28, 28, 1)
x_test = x_test[..., np.newaxis]    # shape: (10000, 28, 28, 1)

# Keep the same ankle boot throughout the initial visualizations.
image_index = 0
image_batch = x_test[image_index:image_index + 1]  # keep batch axis: (1, 28, 28, 1)
flat_image = image_batch.reshape(1, -1)

print("Image shape:", image_batch.shape)
print("Flattened shape:", flat_image.shape)
print("Clothing category:", class_names[int(y_test[image_index])])
```

## Matches Section 1: Flattening removes the 2D row-and-column layout

Flattening preserves all 784 pixel intensities, but adjacent rows and columns are no longer grouped as local 2D neighborhoods by the input structure.

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].imshow(image_batch[0, :, :, 0], cmap="gray")
axes[0].set_title("2D Image: 28 × 28")
axes[0].axis("off")

axes[1].plot(flat_image[0], linewidth=1)
axes[1].set_title("Same 784 pixels flattened into a 1D vector")
axes[1].set_xlabel("Position in vector")
axes[1].set_ylabel("Pixel intensity")
plt.tight_layout()
plt.show()
```

## Matches Section 3: Traditional hand-crafted edge filters on a grayscale image

Apply vertical-edge and horizontal-edge 3 × 3 kernels to a grayscale photograph of a bicycle to see how different manually chosen kernels emphasize different patterns.

```python
from PIL import Image
import requests
from io import BytesIO
from scipy.signal import convolve2d

try:
    image_url = (
        "https://raw.githubusercontent.com/AI-Learning-Repo/Neural-Networks/"
        "refs/heads/week4/material/img/Left_side_of_Flying_Pigeon.jpg"
    )
    response = requests.get(image_url, timeout=15)
    response.raise_for_status()
    img_pil = Image.open(BytesIO(response.content)).convert("RGB")
    img_pil.thumbnail((256, 256))
    img_rgb = np.array(img_pil).astype(np.float32) / 255.0
    img_gray = np.array(img_pil.convert("L")).astype(np.float32) / 255.0
    print(f"Loaded bicycle image: RGB {img_rgb.shape}, Grayscale {img_gray.shape}")
except Exception as exc:
    print(f"Could not download image ({exc}); using synthetic fallback.")
    img_gray = np.zeros((128, 128), dtype=np.float32)
    img_gray[24:104, 24:104] = 1.0
    img_rgb = np.repeat(img_gray[..., np.newaxis], 3, axis=-1)

filter_vertical = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

filter_horizontal = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# Flip kernels for strict mathematical convolution (optional for edge magnitude)
vertical_edges = convolve2d(img_gray, np.flipud(np.fliplr(filter_vertical)), mode="valid")
horizontal_edges = convolve2d(img_gray, np.flipud(np.fliplr(filter_horizontal)), mode="valid")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(img_gray, cmap="gray")
axes[0].set_title(f"Grayscale input ({img_gray.shape[0]}×{img_gray.shape[1]})")
axes[1].imshow(np.abs(vertical_edges), cmap="gray")
axes[1].set_title("Vertical-edge filter response")
axes[2].imshow(np.abs(horizontal_edges), cmap="gray")
axes[2].set_title("Horizontal-edge filter response")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
```

## Matches Section 5: Load the trained Fashion-MNIST CNN and inspect its summary

Rather than choosing kernels manually, a CNN learns filter weights from data. Load the pretrained Fashion-MNIST model and inspect its prediction on the ankle boot.

```python
model_candidates = [
    Path("fashion_mnist_cnn.keras"),
    Path("assets/fashion_mnist_cnn.keras"),
    Path("../../assets/fashion_mnist_cnn.keras"),
]
model_path = next((p for p in model_candidates if p.is_file()), None)

if model_path is None:
    model_url = (
        "https://github.com/AI-Learning-Repo/Neural-Networks/"
        "raw/refs/heads/week4/material/assets/fashion_mnist_cnn.keras"
    )
    model_path = Path("fashion_mnist_cnn.keras")
    print("Downloading fashion_mnist_cnn.keras...")
    urllib.request.urlretrieve(model_url, model_path)

trained_model = keras.models.load_model(model_path, compile=False)
probabilities = trained_model.predict(image_batch, verbose=0)[0]
predicted = int(np.argmax(probabilities))

print("Actual class:", class_names[int(y_test[image_index])])
print("Predicted class:", class_names[predicted])
print("Top probability:", round(float(probabilities[predicted]), 4))
trained_model.summary()
```

## Matches Sections 6–7: Following single 2 × 2 filters on the ankle boot (before ReLU)

Before inspecting all 32 learned filters, observe how two simple 2 × 2 contrast filters slide across the 28 × 28 ankle boot (`padding="VALID"`, `stride=1`) to produce 27 × 27 feature maps containing both positive (red) and negative (blue) responses.

```python
vertical_kernel = np.array([
    [-1.0,  1.0],
    [-1.0,  1.0]
], dtype=np.float32)

horizontal_kernel = np.array([
    [-1.0, -1.0],
    [ 1.0,  1.0]
], dtype=np.float32)

vertical_filter = vertical_kernel.reshape(2, 2, 1, 1)
horizontal_filter = horizontal_kernel.reshape(2, 2, 1, 1)

vertical_map = tf.nn.conv2d(
    image_batch, vertical_filter, strides=1, padding="VALID"
).numpy()[0, :, :, 0]

horizontal_map = tf.nn.conv2d(
    image_batch, horizontal_filter, strides=1, padding="VALID"
).numpy()[0, :, :, 0]

print("Input:", image_batch.shape)
print("Vertical map:", vertical_map.shape)
print("Horizontal map:", horizontal_map.shape)

limit = max(float(np.abs(vertical_map).max()),
            float(np.abs(horizontal_map).max()), 1e-6)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(image_batch[0, :, :, 0], cmap="gray")
axes[0].set_title("Ankle boot (28 × 28)")
axes[1].imshow(vertical_map, cmap="coolwarm", vmin=-limit, vmax=limit)
axes[1].set_title("2×2 Vertical contrast (27 × 27)")
im = axes[2].imshow(horizontal_map, cmap="coolwarm", vmin=-limit, vmax=limit)
axes[2].set_title("2×2 Horizontal contrast (27 × 27)")
for ax in axes:
    ax.axis("off")
fig.colorbar(im, ax=axes[1:], shrink=0.75)
plt.show()
```

## Matches Section 8: All 32 learned filters in `Conv_1`

Extract the trained `Conv_1` kernel weights `(2, 2, 1, 32)` and inspect the 32 resulting feature maps `(27, 27, 32)`. Different learned filters highlight different parts of the ankle boot, and some may produce zero response everywhere after ReLU.

```python
conv1 = trained_model.get_layer("Conv_1")
W1, b1 = conv1.get_weights()
print("First-layer kernel bank shape:", W1.shape)

# Compute responses BEFORE ReLU and verify against Conv_1 output AFTER ReLU.
pre_relu_1 = tf.nn.bias_add(
    tf.nn.conv2d(image_batch, W1, strides=1, padding="VALID"), b1
).numpy()[0]
post_relu_1 = np.maximum(pre_relu_1, 0)

conv1_probe = keras.Model(trained_model.inputs, conv1.output)
actual_conv1 = conv1_probe.predict(image_batch, verbose=0)[0]
np.testing.assert_allclose(post_relu_1, actual_conv1, rtol=1e-4, atol=1e-5)
print("Feature-map tensor shape:", post_relu_1.shape)

# Plot all 32 learned feature maps from Conv_1.
fig, axes = plt.subplots(4, 8, figsize=(14, 7.5))
for ch, ax in enumerate(axes.flat):
    ax.imshow(post_relu_1[:, :, ch], cmap="viridis")
    ax.set_title(f"Filter {ch}", fontsize=8)
    ax.axis("off")
plt.suptitle("All 32 feature maps produced by Conv_1 (27 × 27 × 32)", fontsize=12)
plt.tight_layout()
plt.show()
```

------


# Part 2 Visualizations: What Really Happens Inside a CNN?


## Setup: Load Fashion-MNIST and the trained CNN

If an import is missing in Colab, run `%pip install numpy matplotlib tensorflow scikit-learn` in a notebook cell first.

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import urllib.request

keras.utils.set_random_seed(42)

(x_train_raw, y_train), (x_test_raw, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train_raw.astype("float32")[..., np.newaxis] / 255.0
x_test = x_test_raw.astype("float32")[..., np.newaxis] / 255.0

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

image_index = 0
image_batch = x_test[image_index:image_index + 1]

model_candidates = [
    Path("fashion_mnist_cnn.keras"),
    Path("assets/fashion_mnist_cnn.keras"),
    Path("../../assets/fashion_mnist_cnn.keras"),
]
model_path = next((p for p in model_candidates if p.is_file()), None)

if model_path is None:
    model_url = (
        "https://github.com/AI-Learning-Repo/Neural-Networks/"
        "raw/refs/heads/week4/material/assets/fashion_mnist_cnn.keras"
    )
    model_path = Path("fashion_mnist_cnn.keras")
    print("Downloading fashion_mnist_cnn.keras...")
    urllib.request.urlretrieve(model_url, model_path)

trained_model = keras.models.load_model(model_path, compile=False)
conv1 = trained_model.get_layer("Conv_1")
conv2 = trained_model.get_layer("Conv_2")

probe = keras.Model(
    trained_model.inputs,
    outputs=[conv1.output, conv2.input, conv2.output]
)
first_maps, pooled_input, second_maps = probe.predict(image_batch, verbose=0)
print("Conv_1 output:", first_maps.shape)
print("After first MaxPool:", pooled_input.shape)
print("Conv_2 output:", second_maps.shape)
```

## Matches Sections 12–13: Visualizing the 8 × 8 convolution calculation from scratch

This reproduces the exact 8 × 8 image and 3 × 3 horizontal-edge kernel from Section 12 and displays all numerical values in the resulting 6 × 6 feature map.

```python
image_8x8 = np.array([
    [60, 58, 60, 60, 60, 60, 60, 52],
    [68, 60, 60, 68, 68, 52, 76, 76],
    [76, 44, 60, 60, 68, 52, 60, 52],
    [68, 68, 76, 76, 68, 52, 60, 44],
    [92, 84, 84, 84, 76, 44, 60, 44],
    [76, 68, 84, 76, 76, 52, 60, 52],
    [68, 60, 60, 76, 84, 52, 84, 60],
    [60, 60, 68, 68, 68, 52, 76, 68]
])

kernel_3x3 = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

h, w = image_8x8.shape
kh, kw = kernel_3x3.shape
feature_map_6x6 = np.zeros((h - kh + 1, w - kw + 1), dtype=int)

for r in range(h - kh + 1):
    for c in range(w - kw + 1):
        feature_map_6x6[r, c] = np.sum(image_8x8[r:r + kh, c:c + kw] * kernel_3x3)

print("6 × 6 Feature map:\n", feature_map_6x6)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(image_8x8, cmap="gray", vmin=30, vmax=100)
axes[0].set_title("8 × 8 input image")
for r in range(8):
    for c in range(8):
        axes[0].text(c, r, image_8x8[r, c], ha="center", va="center",
                     color="black" if image_8x8[r, c] > 65 else "white", fontsize=8)

axes[1].imshow(kernel_3x3, cmap="coolwarm", vmin=-2, vmax=2)
axes[1].set_title("3 × 3 kernel")
for r in range(3):
    for c in range(3):
        axes[1].text(c, r, kernel_3x3[r, c], ha="center", va="center", fontsize=11)

limit = np.abs(feature_map_6x6).max()
axes[2].imshow(feature_map_6x6, cmap="coolwarm", vmin=-limit, vmax=limit)
axes[2].set_title("6 × 6 feature map")
for r in range(6):
    for c in range(6):
        axes[2].text(c, r, feature_map_6x6[r, c], ha="center", va="center", fontsize=9)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()
```

## Matches Section 14: Stride 1 vs. Stride 2

Stride controls how many pixels the filter jumps between positions. Stride 2 evaluates the same filter at fewer positions, producing a lower-resolution feature map.

```python
vertical_kernel = np.array([
    [-1.0,  1.0],
    [-1.0,  1.0]
], dtype=np.float32)
vertical_filter = vertical_kernel.reshape(2, 2, 1, 1)

valid_s1 = tf.nn.conv2d(image_batch, vertical_filter, strides=1, padding="VALID").numpy()[0, :, :, 0]
valid_s2 = tf.nn.conv2d(image_batch, vertical_filter, strides=2, padding="VALID").numpy()[0, :, :, 0]

print("Stride 1 output:", valid_s1.shape)
print("Stride 2 output:", valid_s2.shape)

limit = max(float(np.abs(valid_s1).max()), float(np.abs(valid_s2).max()), 1e-6)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(image_batch[0, :, :, 0], cmap="gray")
axes[0].set_title("Input: 28 × 28")
axes[1].imshow(valid_s1, cmap="coolwarm", vmin=-limit, vmax=limit)
axes[1].set_title("2×2 filter, stride 1: 27 × 27")
im = axes[2].imshow(valid_s2, cmap="coolwarm", vmin=-limit, vmax=limit)
axes[2].set_title("Same filter, stride 2: 14 × 14")
for ax in axes:
    ax.axis("off")
fig.colorbar(im, ax=axes[1:], shrink=0.75)
plt.show()
```

## Matches Section 15: Padding (`VALID` vs. `SAME`) and boundary placement

Compare `VALID` (no padding) with `SAME` padding at stride 1, and verify where TensorFlow places the single added row and column for a 2 × 2 kernel.

```python
same_s1 = tf.nn.conv2d(image_batch, vertical_filter, strides=1, padding="SAME").numpy()[0, :, :, 0]

# Reproduce TensorFlow SAME explicitly for a 2 × 2 kernel at stride 1:
# one row of zeros at the bottom and one column of zeros on the right.
explicitly_padded = tf.pad(image_batch, [[0, 0], [0, 1], [0, 1], [0, 0]])
checked_same = tf.nn.conv2d(explicitly_padded, vertical_filter, strides=1, padding="VALID").numpy()[0, :, :, 0]
np.testing.assert_allclose(checked_same, same_s1, rtol=1e-6, atol=1e-6)
print("VALID shape:", valid_s1.shape, "| SAME shape:", same_s1.shape)
print("Explicit bottom/right zero-padding matches SAME:", np.allclose(checked_same, same_s1))

limit = max(float(np.abs(valid_s1).max()), float(np.abs(same_s1).max()), 1e-6)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(image_batch[0, :, :, 0], cmap="gray")
axes[0].set_title("Input: 28 × 28")
axes[1].imshow(valid_s1, cmap="coolwarm", vmin=-limit, vmax=limit)
axes[1].set_title("VALID: 27 × 27")
im = axes[2].imshow(same_s1, cmap="coolwarm", vmin=-limit, vmax=limit)
axes[2].set_title("SAME: 28 × 28")
for ax in axes:
    ax.axis("off")
fig.colorbar(im, ax=axes[1:], shrink=0.75)
plt.show()
```

## Matches Section 16: Pooling — omitted borders, Max vs. Average, and shift sensitivity

### 16a. Omitted border on an odd-sized feature map (`5 × 5` $\rightarrow$ `2 × 2`, just like `27 × 27` $\rightarrow$ `13 × 13`)

```python
toy_grid = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
toy_pooled = tf.nn.max_pool2d(
    toy_grid[np.newaxis, :, :, np.newaxis], ksize=2, strides=2, padding="VALID"
).numpy()[0, :, :, 0]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(toy_grid, cmap="Blues", vmin=0, vmax=25)
axes[0].set_title("5 × 5 input (row/col 4 omitted)")
for r in range(5):
    for c in range(5):
        axes[0].text(c, r, int(toy_grid[r, c]), ha="center", va="center")
axes[0].axhline(3.5, color="red", linewidth=2)
axes[0].axvline(3.5, color="red", linewidth=2)

axes[1].imshow(toy_pooled, cmap="Blues", vmin=0, vmax=25)
axes[1].set_title("2 × 2 MaxPool output")
for r in range(2):
    for c in range(2):
        axes[1].text(c, r, int(toy_pooled[r, c]), ha="center", va="center")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()
```

### 16b. Max pooling vs. Average pooling on a 4 × 4 example and on a real `Conv_1` feature map

```python
pool_example = np.array([
    [1, 2, 0, 0],
    [3, 9, 1, 0],
    [0, 0, 2, 3],
    [0, 0, 4, 5]
], dtype=np.float32)
pool_batch = pool_example[np.newaxis, :, :, np.newaxis]

example_max = tf.nn.max_pool2d(pool_batch, ksize=2, strides=2, padding="VALID").numpy()[0, :, :, 0]
example_avg = tf.nn.avg_pool2d(pool_batch, ksize=2, strides=2, padding="VALID").numpy()[0, :, :, 0]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
for ax, values, title in zip(
    axes,
    [pool_example, example_max, example_avg],
    ["Input: 4 × 4", "MaxPool: 2 × 2", "AveragePool: 2 × 2"]
):
    ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=9)
    ax.set_title(title)
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            ax.text(c, r, f"{values[r, c]:g}", ha="center", va="center", color="black")
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

# Apply both pooling methods to an actual Conv_1 learned feature map.
max_maps = tf.nn.max_pool2d(first_maps, ksize=2, strides=2, padding="VALID").numpy()
avg_maps = tf.nn.avg_pool2d(first_maps, ksize=2, strides=2, padding="VALID").numpy()
np.testing.assert_allclose(max_maps, pooled_input, rtol=1e-5, atol=1e-6)

channel = int(np.argmax(first_maps[0].std(axis=(0, 1))))
before_pool = first_maps[0, :, :, channel]
after_max = max_maps[0, :, :, channel]
after_avg = avg_maps[0, :, :, channel]

upper = max(float(before_pool.max()), float(after_max.max()), float(after_avg.max()), 1e-6)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, values, title in zip(
    axes,
    [before_pool, after_max, after_avg],
    [f"Conv_1 ch {channel}: 27 × 27", "Max pooling: 13 × 13", "Average pooling: 13 × 13"]
):
    im = ax.imshow(values, cmap="viridis", vmin=0, vmax=upper)
    ax.set_title(title)
    ax.axis("off")
fig.colorbar(im, ax=axes, shrink=0.75)
plt.show()
```

### 16c. Local shift tolerance in max pooling

A 1-pixel shift inside the same 2 × 2 window leaves the pooled output unchanged, whereas a 1-pixel shift across a window boundary moves the pooled peak.

```python
def single_response(column):
    grid = np.zeros((4, 4), dtype=np.float32)
    grid[0, column] = 9.0
    return grid

shifted_inputs = [single_response(c) for c in (0, 1, 2)]
shifted_outputs = [
    tf.nn.max_pool2d(g[np.newaxis, :, :, np.newaxis], ksize=2, strides=2, padding="VALID").numpy()[0, :, :, 0]
    for g in shifted_inputs
]

print("Same output for columns 0 and 1:", np.array_equal(shifted_outputs[0], shifted_outputs[1]))
print("Same output for columns 1 and 2:", np.array_equal(shifted_outputs[1], shifted_outputs[2]))

fig, axes = plt.subplots(3, 2, figsize=(7, 9))
for row, (inp, out) in enumerate(zip(shifted_inputs, shifted_outputs)):
    axes[row, 0].imshow(inp, cmap="YlOrRd", vmin=0, vmax=9)
    axes[row, 0].set_title(f"Peak at column {row}")
    axes[row, 1].imshow(out, cmap="YlOrRd", vmin=0, vmax=9)
    axes[row, 1].set_title("After 2 × 2 max pooling")
    for ax, values in ((axes[row, 0], inp), (axes[row, 1], out)):
        for r in range(values.shape[0]):
            for c in range(values.shape[1]):
                ax.text(c, r, f"{values[r, c]:g}", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
plt.show()
```

## Matches Section 17: Combining padding, stride, and pooling

Compare all four combinations of `VALID`/`SAME` padding and stride `1`/`2` followed by 2 × 2 max pooling on the same 28 × 28 image.

```python
settings = [("VALID", 1), ("SAME", 1), ("VALID", 2), ("SAME", 2)]
comparison = []

for padding, stride in settings:
    response = tf.nn.conv2d(image_batch, vertical_filter, strides=stride, padding=padding).numpy()
    pooled = tf.nn.max_pool2d(response, ksize=2, strides=2, padding="VALID").numpy()
    comparison.append((padding, stride, response, pooled))
    print(f"Conv padding={padding:5s}, stride={stride}: conv={response.shape[1:3]}, pooled={pooled.shape[1:3]}")

scale = max(*(float(np.abs(item[2]).max()) for item in comparison), 1e-6)
fig, axes = plt.subplots(2, 2, figsize=(9, 8))
for ax, (padding, stride, response, pooled) in zip(axes.flat, comparison):
    ax.imshow(response[0, :, :, 0], cmap="coolwarm", vmin=-scale, vmax=scale)
    ax.set_title(
        f"Conv {padding}, stride {stride}\n"
        f"{response.shape[1]} × {response.shape[2]} → pool {pooled.shape[1]} × {pooled.shape[2]}"
    )
    ax.axis("off")
plt.tight_layout()
plt.show()
```

## Matches Section 18: How `Conv_2` combines all 32 incoming feature maps

First compare six active maps from `Conv_1` and `Conv_2`, then decompose **one** `Conv_2` output filter (`2 × 2 × 32`) to inspect how individual incoming channels combine before ReLU.

```python
# Part A: Conv_1 vs Conv_2 feature maps
fig, axes = plt.subplots(2, 6, figsize=(15, 6))
for row, maps in enumerate((first_maps[0], second_maps[0])):
    selected = np.argsort(maps.std(axis=(0, 1)))[-6:][::-1]
    for col, ch in enumerate(selected):
        axes[row, col].imshow(maps[:, :, ch], cmap="viridis")
        axes[row, col].set_title(f"Conv_{row + 1}: {ch}")
        axes[row, col].axis("off")
plt.tight_layout()
plt.show()

# Part B: Decompose one Conv_2 output filter across all 32 input channels
W2, b2 = conv2.get_weights()
out_channel = int(np.argmax(second_maps[0].mean(axis=(0, 1))))
filter_3d = W2[:, :, :, out_channel]

height, width = second_maps.shape[1:3]
contributions = np.zeros((height, width, pooled_input.shape[-1]), dtype=np.float32)
for r in range(height):
    for c in range(width):
        patch_3d = pooled_input[0, r:r + 2, c:c + 2, :]
        contributions[r, c, :] = np.sum(patch_3d * filter_3d, axis=(0, 1))

combined = contributions.sum(axis=-1) + b2[out_channel]
activated = np.maximum(combined, 0)
np.testing.assert_allclose(activated, second_maps[0, :, :, out_channel], rtol=1e-4, atol=1e-5)

strength = np.abs(contributions).mean(axis=(0, 1))
top4 = np.argsort(strength)[-4:][::-1]
scale = max(float(np.abs(contributions[:, :, top4]).max()), float(np.abs(combined).max()), 1e-6)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, ch in zip(axes.flat[:4], top4):
    ax.imshow(contributions[:, :, ch], cmap="coolwarm", vmin=-scale, vmax=scale)
    ax.set_title(f"Input channel {ch}")
    ax.axis("off")
axes.flat[4].imshow(combined, cmap="coolwarm", vmin=-scale, vmax=scale)
axes.flat[4].set_title(f"All 32 channels + bias (Filter {out_channel})")
axes.flat[5].imshow(activated, cmap="coolwarm", vmin=-scale, vmax=scale)
axes.flat[5].set_title("After ReLU")
for ax in axes.flat[4:]:
    ax.axis("off")
plt.tight_layout()
plt.show()
```

## Matches Section 19: What ReLU preserves and hides (before vs. after ReLU)

Inspect two revealing `Conv_1` filters on the ankle boot: filter 2 (suppressed everywhere) and filter 12 (positive regions kept, negative regions zeroed).

```python
W1, b1 = conv1.get_weights()
pre_relu_1 = tf.nn.bias_add(
    tf.nn.conv2d(image_batch, W1, strides=1, padding="VALID"), b1
).numpy()[0]
post_relu_1 = np.maximum(pre_relu_1, 0)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for row, channel in enumerate((2, 12)):
    before = pre_relu_1[:, :, channel]
    after = post_relu_1[:, :, channel]
    limit = max(float(np.abs(before).max()), 1e-6)

    print(f"Filter {channel} — min/max before ReLU: {before.min():.4f} / {before.max():.4f} | "
          f"zero fraction after ReLU: {np.mean(after == 0):.4f}")

    axes[row, 0].imshow(image_batch[0, :, :, 0], cmap="gray")
    axes[row, 0].set_title("Original ankle boot")
    axes[row, 1].imshow(before, cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[row, 1].set_title(f"Filter {channel}: before ReLU")
    axes[row, 2].imshow(after, cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[row, 2].set_title(f"Filter {channel}: after ReLU")
    for ax in axes[row]:
        ax.axis("off")
plt.tight_layout()
plt.show()
```

## Matches Section 20: Train and evaluate the complete CNN (curves, mistakes, confusion matrix, F1)

Build the full 302,026-parameter CNN, train on Fashion-MNIST, and examine both overall metrics (accuracy, macro/weighted F1) and class-level errors.

```python
from sklearn.metrics import classification_report, f1_score

keras.utils.set_random_seed(42)

inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, (2, 2), name="demo_conv1")(inputs)
x = layers.ReLU(name="demo_relu1")(x)
x = layers.MaxPooling2D((2, 2), name="demo_pool1")(x)
x = layers.Conv2D(32, (2, 2), name="demo_conv2")(x)
x = layers.ReLU(name="demo_relu2")(x)
x = layers.MaxPooling2D((2, 2), name="demo_pool2")(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)

demo_model = keras.Model(inputs, outputs, name="fashion_cnn_demo")
demo_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = demo_model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.2,
    verbose=2
)

# 1. Training and validation curves
epochs = np.arange(1, len(history.history["loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(epochs, history.history["loss"], marker="o", label="Train")
axes[0].plot(epochs, history.history["val_loss"], marker="o", label="Validation")
axes[0].set_title("Loss")
axes[1].plot(epochs, history.history["accuracy"], marker="o", label="Train")
axes[1].plot(epochs, history.history["val_accuracy"], marker="o", label="Validation")
axes[1].set_title("Accuracy")
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.2)
    ax.legend()
plt.tight_layout()
plt.show()

# 2. Test evaluation & F1 scores (assumes integer-encoded y_test 0–9)
test_loss, test_accuracy = demo_model.evaluate(x_test, y_test, verbose=0)
all_probs = demo_model.predict(x_test, verbose=0)
predictions = np.argmax(all_probs, axis=1)
wrong_indices = np.flatnonzero(predictions != y_test)

print(f"Test accuracy: {test_accuracy:.4f} | Test loss: {test_loss:.4f} | Misclassified: {len(wrong_indices)}")
print(classification_report(y_test, predictions, target_names=class_names, digits=3))
print("Macro F1:", round(float(f1_score(y_test, predictions, average="macro")), 4))
print("Weighted F1:", round(float(f1_score(y_test, predictions, average="weighted")), 4))

# 3. Misclassified examples & confusion matrix
chosen = np.random.default_rng(42).choice(wrong_indices, min(12, len(wrong_indices)), replace=False)
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for ax in axes.flat:
    ax.axis("off")
for ax, idx in zip(axes.flat, chosen):
    ax.imshow(x_test[idx, :, :, 0], cmap="gray")
    ax.set_title(f"True: {class_names[int(y_test[idx])]}\nPred: {class_names[int(predictions[idx])]}")
plt.tight_layout()
plt.show()

counts = tf.math.confusion_matrix(y_test, predictions, num_classes=10).numpy()
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(counts, cmap="Blues")
ax.set_xticks(range(10), class_names, rotation=55, ha="right")
ax.set_yticks(range(10), class_names)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
for i in range(10):
    for j in range(10):
        ax.text(j, i, counts[i, j], ha="center", va="center",
                fontsize=7, color="white" if counts[i, j] > counts.max() * 0.5 else "black")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()
```

----

# Part 3 Visualizations: From Grayscale to Color Images


## Setup: Load a color photograph (RGB)

If an import is missing in Colab, run `%pip install numpy matplotlib tensorflow pillow requests` in a notebook cell first.

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path
import urllib.request

keras.utils.set_random_seed(42)

try:
    image_url = (
        "https://raw.githubusercontent.com/AI-Learning-Repo/Neural-Networks/"
        "refs/heads/week4/material/img/Left_side_of_Flying_Pigeon.jpg"
    )
    response = requests.get(image_url, timeout=15)
    response.raise_for_status()
    img_pil = Image.open(BytesIO(response.content)).convert("RGB")
    img_pil.thumbnail((256, 256))
    img_rgb = np.array(img_pil).astype(np.float32) / 255.0
    print("Loaded RGB image with shape:", img_rgb.shape)
except Exception as exc:
    print(f"Could not download image ({exc}); using synthetic RGB fallback.")
    img_rgb = np.zeros((128, 128, 3), dtype=np.float32)
    img_rgb[20:80, 20:80, 0] = 0.9   # red square
    img_rgb[40:100, 50:110, 1] = 0.8 # green square
    img_rgb[60:115, 30:90, 2] = 0.9  # blue square
```

## Matches Section 21: Decomposing an RGB image into its three color channels

Each pixel in an RGB image stores three numbers—Red, Green, and Blue intensities—stacked along the channel axis `(height, width, 3)`. Display each channel as an intensity matrix (`0` = dark, `1` = bright).

```python
fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
axes[0].imshow(img_rgb)
axes[0].set_title(f"RGB image\n{img_rgb.shape}")

channel_names = ["Red channel (R)", "Green channel (G)", "Blue channel (B)"]
for idx, (ax, title) in enumerate(zip(axes[1:], channel_names)):
    ax.imshow(img_rgb[:, :, idx], cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(f"{title}\n{img_rgb[:, :, idx].shape}")

for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
```

## Matches Sections 22–23: One RGB filter combines three input channels into one feature map

### 23a. Exact numerical verification of the Section 23 calculation

Verify the 2 × 2 × 3 calculation (`4 + 2 + 0 + 1 = 7`) using both NumPy and `tf.nn.conv2d`.

```python
rgb_patch = np.zeros((1, 2, 2, 3), dtype=np.float32)
rgb_patch[0, :, :, 0] = [[2, 1], [0, 2]]  # Red patch
rgb_patch[0, :, :, 1] = [[1, 0], [2, 1]]  # Green patch
rgb_patch[0, :, :, 2] = [[1, 2], [0, 1]]  # Blue patch

rgb_kernel = np.zeros((2, 2, 3, 1), dtype=np.float32)
rgb_kernel[:, :, 0, 0] = [[ 1, 0], [0, 1]]  # Red weights
rgb_kernel[:, :, 1, 0] = [[ 0, 1], [1, 0]]  # Green weights
rgb_kernel[:, :, 2, 0] = [[-1, 0], [0, 1]]  # Blue weights
bias = np.array([1.0], dtype=np.float32)

manual_sum = np.sum(rgb_patch[0] * rgb_kernel[:, :, :, 0]) + bias[0]
tf_output = tf.nn.bias_add(
    tf.nn.conv2d(rgb_patch, rgb_kernel, strides=1, padding="VALID"), bias
).numpy()

print("Manual RGB convolution result:", manual_sum)
print("TensorFlow output shape:", tf_output.shape, "| value:", tf_output[0, 0, 0, 0])
```

### 23b. Apply one 3 × 3 × 3 filter to the full color photograph

Even though the input has 3 channels, **one** filter produces **one** 2D feature map.

```python
single_rgb_filter = np.zeros((3, 3, 3, 1), dtype=np.float32)
sobel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
# Detect vertical edges across all three color channels equally.
for c in range(3):
    single_rgb_filter[:, :, c, 0] = sobel_v / 3.0

rgb_batch = img_rgb[np.newaxis, ...]  # shape: (1, H, W, 3)
one_feature_map = tf.nn.conv2d(rgb_batch, single_rgb_filter, strides=1, padding="VALID").numpy()

print("RGB input batch shape:", rgb_batch.shape)
print("Filter shape:", single_rgb_filter.shape)
print("Single output feature map shape:", one_feature_map.shape)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img_rgb)
axes[0].set_title(f"RGB input {img_rgb.shape}")
axes[1].imshow(np.abs(one_feature_map[0, :, :, 0]), cmap="gray")
axes[1].set_title(f"1 filter output {one_feature_map.shape[1:]}")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
```

## Matches Sections 24–26: Multiple RGB filters, shapes, padding, and stride in Keras

Run the Section 26 shape-verification experiment on `(1, 28, 28, 3)`, check the Section 25 padding/stride output shapes, and visualize 6 random (untrained) `Conv2D` feature maps on the color bicycle image.

```python
# 1. Section 26 shape check: 28 × 28 × 3 -> 27 × 27 × 32 (416 params) -> 13 × 13 × 32
rgb_sample = tf.random.uniform((1, 28, 28, 3))
conv = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu")
feature_maps = conv(rgb_sample)
pooled = keras.layers.MaxPooling2D(pool_size=(2, 2))(feature_maps)

print("Input:", rgb_sample.shape)
print("Kernel weights:", conv.kernel.shape)
print("Output:", feature_maps.shape)
print("Parameters:", conv.count_params())
print("After pooling:", pooled.shape)

# 2. Section 25 table check: 28 × 28 × 3 with 32 filters of size 3 × 3
print("\nSection 25 shape checks (3 × 3 kernels, 32 filters):")
for pad, s in [("valid", 1), ("same", 1), ("valid", 2)]:
    out = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=s, padding=pad)(rgb_sample)
    print(f"  padding={pad:5s}, stride={s} -> {tuple(out.shape[1:])}")

# 3. Apply a 6-filter Conv2D layer to the real color bicycle image (untrained weights)
rgb_conv = keras.layers.Conv2D(6, kernel_size=(3, 3), activation="relu")
bike_maps = rgb_conv(rgb_batch).numpy()[0]

fig, axes = plt.subplots(1, 7, figsize=(15, 2.8))
axes[0].imshow(img_rgb)
axes[0].set_title("RGB input")
axes[0].axis("off")
for i in range(6):
    axes[i + 1].imshow(bike_maps[:, :, i], cmap="viridis")
    axes[i + 1].set_title(f"Filter {i}")
    axes[i + 1].axis("off")
plt.suptitle("RGB image passed through 6 untrained Conv2D filters (6 output channels)", fontsize=11)
plt.tight_layout()
plt.show()
```

## Matches Section 27: Classifier heads — Flatten vs. Global Average Pooling (GAP)

Load the saved Fashion-MNIST CNN to compare **Flatten** (keeps every spatial value) against **Global Average Pooling** (takes one mean per channel) on the `6 × 6 × 32` tensor entering the classifier head.

```python
(_, _), (x_test_raw, _) = keras.datasets.fashion_mnist.load_data()
ankle_boot = x_test_raw[0:1].astype("float32")[..., np.newaxis] / 255.0

model_candidates = [
    Path("fashion_mnist_cnn.keras"),
    Path("assets/fashion_mnist_cnn.keras"),
    Path("../../assets/fashion_mnist_cnn.keras"),
]
model_path = next((p for p in model_candidates if p.is_file()), None)
if model_path is None:
    model_url = (
        "https://github.com/AI-Learning-Repo/Neural-Networks/"
        "raw/refs/heads/week4/material/assets/fashion_mnist_cnn.keras"
    )
    model_path = Path("fashion_mnist_cnn.keras")
    urllib.request.urlretrieve(model_url, model_path)

trained_model = keras.models.load_model(model_path, compile=False)
pooling_layers = [l for l in trained_model.layers if isinstance(l, keras.layers.MaxPooling2D)]
backbone_probe = keras.Model(trained_model.inputs, pooling_layers[-1].output)

spatial_features = backbone_probe.predict(ankle_boot, verbose=0)  # shape: (1, 6, 6, 32)
flat_vector = keras.layers.Flatten()(spatial_features)
gap_vector = keras.layers.GlobalAveragePooling2D()(spatial_features)

np.testing.assert_allclose(gap_vector.numpy(), spatial_features.mean(axis=(1, 2)), rtol=1e-5, atol=1e-6)
print("Spatial tensor before classifier:", spatial_features.shape)
print("Flatten vector shape:", flat_vector.shape, "-> Dense(256) params:", int(flat_vector.shape[-1]) * 256 + 256)
print("GAP vector shape:    ", gap_vector.shape,  "-> Dense(256) params:", int(gap_vector.shape[-1]) * 256 + 256)
```


<!-- 
---

## Looking ahead (Next Colab): Reusing learned filters with Transfer Learning

The next Colab notebook applies CNNs to color images and transfer learning. This short structural check shows how a pretrained convolutional backbone can be **frozen** (training only a new classifier head) or **selectively fine-tuned** (unfreezing later convolutional layers with a smaller learning rate).

```python
transfer_source = keras.models.clone_model(trained_model)
transfer_source.set_weights(trained_model.get_weights())
src_pools = [l for l in transfer_source.layers if isinstance(l, keras.layers.MaxPooling2D)]

backbone = keras.Model(transfer_source.inputs, src_pools[-1].output, name="demo_transfer_backbone")
backbone.trainable = False  # Stage 1: freeze entire backbone

target_inputs = keras.Input(shape=(28, 28, 1))
target_features = backbone(target_inputs, training=False)
target_vector = keras.layers.GlobalAveragePooling2D()(target_features)
target_outputs = keras.layers.Dense(5, activation="softmax")(target_vector)
transfer_model = keras.Model(target_inputs, target_outputs)

def scalar_parameter_count(weights):
    return sum(int(np.prod(w.shape)) for w in weights)

transfer_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
print("Stage 1 (Frozen backbone) — trainable scalars:",
      scalar_parameter_count(transfer_model.trainable_weights),
      "| non-trainable scalars:",
      scalar_parameter_count(transfer_model.non_trainable_weights))

# Stage 2: unfreeze only the last convolutional layer for fine-tuning
backbone.trainable = True
for layer in backbone.layers:
    layer.trainable = False
conv_layers = [l for l in backbone.layers if isinstance(l, keras.layers.Conv2D)]
conv_layers[-1].trainable = True

transfer_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
print("Stage 2 (Fine-tuning", conv_layers[-1].name + ") — trainable scalars:",
      scalar_parameter_count(transfer_model.trainable_weights),
      "| non-trainable scalars:",
      scalar_parameter_count(transfer_model.non_trainable_weights))
```

-->
