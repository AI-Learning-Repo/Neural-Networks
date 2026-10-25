# Python setup for Week 2

The demos and activities provide plotting code in expandable sections. Open a code section, run the supplied code, observe the plot, and explain the result. Writing plotting code from scratch is not assessed. Hand calculations are optional practice; worked examples remain available for checking the reasoning.

## Google Colab setup

Google Colab is the main environment for these materials. First run the imports in the relevant example. Many libraries are already available. If an import reports `ModuleNotFoundError`, install the missing package in a separate code cell.

| Material | Required libraries |
|---|---|
| Lecture plots | NumPy, Matplotlib |
| Symbolic calculus examples | SymPy |
| Titanic, Auto MPG, and Heart pair programming | NumPy, pandas, Matplotlib, scikit-learn, Keras, TensorFlow |

The package named `scikit-learn` is imported as `sklearn`. Run only the notebook command needed for the missing library or group:

```text
%pip install numpy matplotlib
%pip install sympy
%pip install pandas scikit-learn
%pip install keras tensorflow
```

Paste the required line into a Colab code cell. No upgrade is needed when existing imports work. If installation requests a runtime/session restart, restart it, then rerun the notebook from its setup cell. Loading the datasets requires internet access.

Check lecture dependencies with:

```python
import numpy
import matplotlib
import sympy

print("Lecture libraries imported successfully.")
```

For pair programming, also check:

```python
import pandas
import sklearn
import keras
import tensorflow

print("Pair-programming libraries imported successfully.")
print("Keras backend:", keras.config.backend())
```

The supplied Keras examples assume the TensorFlow backend used in the course. A standard new Colab runtime normally uses this backend.

## Optional local Conda setup

NumPy and Matplotlib support the plots. SymPy supports symbolic differentiation in the calculus review. The short arithmetic simulations also use Python's built-in `math` module, which requires no installation.

Run these commands in a terminal where Conda is available. The activation path below is for the course environment on this computer; use the corresponding course environment path on another computer.

```powershell
conda activate G:\AI-ML-Teacher\AI\NN\.conda\nn-course
conda install numpy matplotlib sympy pandas scikit-learn
```

If Conda requests confirmation, enter `y`. Installation is needed only when a required library is missing.

## Check the active environment

Run the following command in the same terminal:

```powershell
python -c "import sys, numpy, matplotlib, sympy; print('Python:', sys.executable); print('NumPy:', numpy.__version__); print('Matplotlib:', matplotlib.__version__); print('SymPy:', sympy.__version__)"
```

The Python path should point to the activated course environment. Three library version numbers should appear without an import error.

For Jupyter, select the kernel associated with the course environment. Restart the kernel after installing a missing library, then rerun the supplied example.
