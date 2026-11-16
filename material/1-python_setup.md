# Python setup for CNN demonstrations and activities

Google Colab is the primary environment for the CNN materials. A local Jupyter notebook using an appropriate Python environment is an alternative. The activities provide plotting code for running and interpreting; writing Matplotlib code from scratch is not assessed.

## 1. Google Colab

1. Open a Colab notebook and select a Python runtime. A CPU is sufficient for the small examples; a GPU is optional and may accelerate model training.
2. Run the **verification cell** below before installing anything. Colab commonly includes NumPy, Matplotlib, and TensorFlow.
3. Install **only missing packages**. Avoid unnecessary upgrades of packages that already work together.
4. If a package installation requires a runtime restart, restart it and rerun all dependent cells in order.

| Purpose | Python package / tool | Needed for |
|---|---|---|
| Arrays and numerical checks | `numpy` | All CNN activities |
| Plots and feature-map grids | `matplotlib` | All CNN activities |
| CNNs, datasets, training, pooling | `tensorflow` (includes `tf.keras`) | All CNN activities |
| Notebook image display | `IPython` | Optional display cells |
| Keras architecture drawing | `pydot` and **Graphviz system executable** | Optional `keras.utils.plot_model()` |

### Verify the core environment

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("NumPy:", np.__version__)
print("TensorFlow:", tf.__version__)
print("Eager execution:", tf.executing_eagerly())
print("Core CNN libraries imported successfully.")
```

If one of these imports fails, use the relevant command in a **separate Colab code cell**:

```text
%pip install numpy matplotlib
%pip install tensorflow
```

Do not paste the commands into ordinary Python source files. They are notebook commands. A new environment may need an installation; an existing working Colab runtime usually does not.

### Optional architecture diagrams with Graphviz

A Mermaid architecture diagram in the Markdown activity does **not** depend on Graphviz. Graphviz is needed only when running the optional Keras `plot_model()` visualization.

```python
import shutil

try:
    import pydot
    print("pydot:", pydot.__version__)
except ImportError:
    print("pydot is unavailable; Keras architecture plotting is optional.")

print("Graphviz executable:", shutil.which("dot"))
```

If necessary in **Colab only**, install the missing dependencies in separate cells:

```text
%pip install pydot
!apt-get -qq install graphviz
```

The `!apt-get` line is a Linux/Colab command, not a Windows command. A plot that relies on Graphviz may be skipped without preventing model training, model summaries, or the conceptual exercises.

Use a guarded optional architecture plot:

```python
try:
    keras.utils.plot_model(
        model,
        to_file="cnn_architecture.png",
        show_shapes=True,
        show_layer_names=True,
        dpi=100,
    )
    from IPython.display import Image, display
    display(Image(filename="cnn_architecture.png"))
except Exception as exc:
    print("Architecture image unavailable; model.summary() remains usable.")
    print(type(exc).__name__, str(exc)[:250])
```

Here `model` refers to a Keras model already defined in the notebook.

## 2. Datasets, files, and run order

### Start each activity with a clean runtime

Use a separate notebook for each guided activity. Before starting Activity 2 after Activity 1, connect to a fresh Colab runtime or restart the existing session. In local Jupyter, restart the kernel. Opening another notebook file is not a substitute for restarting a kernel that is being reused.

A restart clears the running session's variables, models, and in-memory datasets. Rerun the activity's verification/import cell and data setup afterward. This prevents old variables such as `model`, dataset arrays, or training histories from being mistaken for the current activity's objects and releases memory held by the previous session.

Within an activity, run cells in order. In Activity 2, the frozen-backbone and fine-tuning sections reuse the source model trained earlier in that same activity. If the runtime is restarted midway, rerun those prerequisite cells before continuing; do not assume the trained backbone or its weights remain available in memory.

- **MNIST** is used in CNN Activities 1 and 2. `keras.datasets.mnist.load_data()` downloads the dataset on first use if it is not already cached. The activities do not need an uploaded `.keras` file.
- **Fashion-MNIST** is used in the companion demonstrations; a demonstration inspecting a saved CNN may require the separately supplied model file.


## 3. Optional local Conda / Jupyter environment

From a terminal with the course environment activated, confirm the Python interpreter:

```text
python -c "import sys; print(sys.executable)"
```

Install only missing libraries, using the package manager appropriate to the environment:

```text
python -m pip install numpy matplotlib tensorflow ipython pydot
```

Graphviz is a **separate executable** in addition to the Python package `pydot`. A Conda environment may install it using:

```text
conda install graphviz
```

Select the same environment as the Jupyter notebook kernel, restart the kernel after installation if needed, and run the verification cell again. The optional Graphviz step can be omitted when only the Mermaid diagrams and `model.summary()` are needed.

## 4. Common troubleshooting

| Symptom | Check |
|---|---|
| `ModuleNotFoundError` | Install the missing package in the active notebook environment, then rerun imports. |
| `dot` / Graphviz not found | Check `shutil.which("dot")`; skip optional architecture plotting or install the system executable. |
| `NameError` for a dataset, layer, or history variable | Run earlier code cells in order; do not skip the cell that defines the variable. |
| Input shape mismatch | Confirm CNN input batches are shaped `(batch, 28, 28, 1)`, not `(batch, 28, 28)`. |
| Loss or label mismatch | For integer digit-class labels, use a softmax output with `sparse_categorical_crossentropy`. |
| Transfer-learning class IDs mismatch | When treating digits 5–9 as a separate five-class task, subtract 5 from training targets and add 5 back when displaying predictions. |
| Different training results | Random seeds help reproducibility but do not guarantee identical results on every hardware/software configuration. |


