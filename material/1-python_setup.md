# Python setup for Week 3

Google Colab is the primary environment for the image-classification materials. The supplied plotting code is for running, viewing, and interpreting; writing plotting code from scratch is not assessed. 

## Google Colab setup

Run the imports first. Many required packages are already available in Colab. If an import reports a missing package, install only that package in a separate code cell.

| Material | Required libraries or tools |
|---|---|
| Image arrays and plots | NumPy, Matplotlib |
| Model construction, training, and datasets | TensorFlow/Keras |
| Displaying generated images in a notebook | IPython |
| Keras architecture diagrams | pydot and the Graphviz system executable |

Use the relevant notebook command only when its package is missing:

```text
%pip install numpy matplotlib
%pip install tensorflow
%pip install ipython pydot
```

Architecture diagrams also require the Graphviz executable. If it is missing in Colab, run this in a separate code cell:

```text
!apt-get -qq install graphviz
```

The Graphviz command above is for Colab/Linux, not Windows. If installation requests a runtime/session restart, restart it and rerun the notebook from its setup cell. Existing working packages do not need upgrading.

## Verify the notebook environment

Run this cell before the activities:

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print("TensorFlow:", tf.__version__)
print("TensorFlow executing eagerly:", tf.executing_eagerly())
print("Core image-classification libraries imported successfully.")
```

Before running an architecture-plot cell, also check:

```python
import shutil
import pydot
from IPython.display import Image, display

dot_path = shutil.which("dot")
print("pydot:", pydot.__version__)
print("Graphviz executable:", dot_path)
if dot_path is None:
    print("Install Graphviz before running keras.utils.plot_model().")
```

The examples use TensorFlow's Keras interface, `tf.keras`. MNIST, Fashion-MNIST, and CIFAR-10 download on first use. Subsequent runs can reuse downloaded files in the same environment. A new Colab runtime may need to download them again.

## Optional local Conda setup

Open a Conda-enabled terminal at the course repository root. Activate the course environment before installing missing dependencies:

```powershell
conda activate .\.conda\nn-course
python -c "import sys; print(sys.executable)"
```

The Python path should point inside the course environment. If needed, install the missing packages using the corresponding command:

```powershell
conda install numpy matplotlib
python -m pip install tensorflow
conda install ipython pydot graphviz
```

These are terminal commands, not notebook cells. After installation, select the course environment as the notebook kernel and restart the kernel. Run the verification cells above. The Graphviz executable should be available in the activated environment.
