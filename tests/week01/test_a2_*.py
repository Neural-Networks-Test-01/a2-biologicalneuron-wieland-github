
import os
import nbformat
import numpy as np
import types
import importlib
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

# Path to the student notebook
HERE = Path(__file__).resolve().parent          # tests/week01
ROOT = HERE.parents[1]                          # Projekt-Root
NB_PATH = ROOT / "a2_BiologicalNeuron.ipynb"

def _exec_notebook(path):
    """Execute the notebook and return the global namespace dict used during execution."""
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
    # We'll collect variables by providing a custom 'resources' with a fresh namespace
    # ExecutePreprocessor runs in its own kernel; we'll rely on inspecting results by re-executing definitions here.
    # Instead, after execution, we grab the last cell's outputs; easier is to re-execute and then import via exec of concatenated code cells.
    # Simpler approach: concatenate all code cells and exec in a controlled dict.
    code_cells = [c.source for c in nb.cells if c.cell_type == "code" and c.source.strip()]
    src = "\\n\\n".join(code_cells)
    ns = {}
    exec(compile(src, path, "exec"), ns, ns)
    return ns

def test_notebook_executes():
    ns = _exec_notebook(NB_PATH)
    assert isinstance(ns, dict)
    # Core symbols must exist
    assert "tilting_bucket_model" in ns, "Function tilting_bucket_model must be defined"
    assert "Perceptron" in ns, "Class Perceptron must be defined"

def test_tilting_bucket_model_behavior():
    ns = _exec_notebook(NB_PATH)
    f = ns["tilting_bucket_model"]
    # Call with simple params
    spikes, levels = f(threshold=1.0, leak_rate=0.1, input_current=0.2, time_steps=50)
    # Basic shape checks
    assert len(spikes) == 50 and len(levels) == 50, "Outputs must have length time_steps"
    # Spikes must be 0/1
    assert set(np.unique(spikes)).issubset({0,1}), "Spikes must be binary 0/1"
    # Levels must be finite
    assert np.all(np.isfinite(levels)), "Bucket levels must be finite numbers"

def test_tilting_bucket_model_stochastic():
    ns = _exec_notebook(NB_PATH)
    f = ns.get("tilting_bucket_model_stochastic")
    assert callable(f), "Function tilting_bucket_model_stochastic must be defined"
    # Determinism for same seed / difference for different seeds
    np.random.seed(0)
    s1, l1 = f(1.0, 0.1, 0.2, 100, noise_std=0.05)
    np.random.seed(0)
    s2, l2 = f(1.0, 0.1, 0.2, 100, noise_std=0.05)
    np.random.seed(1)
    s3, l3 = f(1.0, 0.1, 0.2, 100, noise_std=0.05)
    # Same seed -> identical outputs
    assert np.array_equal(np.array(s1), np.array(s2)) and np.allclose(l1, l2), "Stochastic model should be reproducible under fixed seed"
    # Different seed -> likely different outputs
    assert not np.array_equal(np.array(s1), np.array(s3)) or not np.allclose(l1, l3), "Different seeds should usually change the outcome"

def test_tilting_bucket_model_adaptation():
    ns = _exec_notebook(NB_PATH)
    f = ns.get("tilting_bucket_model_adaptation")
    assert callable(f), "Function tilting_bucket_model_adaptation must be defined"
    s, l = f(1.0, 0.1, 0.3, 100, adaptation_rate=0.01)
    assert len(s) == 100 and len(l) == 100
    assert set(np.unique(s)).issubset({0,1})
    # With adaptation, we expect at least one spike to occur for these params
    assert np.sum(s) > 0, "Adaptation model should produce at least one spike for given parameters"

def test_perceptron_init_and_predict_train():
    ns = _exec_notebook(NB_PATH)
    Perceptron = ns["Perceptron"]
    p = Perceptron(input_size=2)  # rely on default learning_rate
    # Attributes
    assert hasattr(p, "weights") and hasattr(p, "bias") and hasattr(p, "learning_rate")
    assert isinstance(p.bias, (int, float))
    assert p.weights.shape == (2,), "Weights shape should match input_size"
    # Default LR is expected to be 0.5 per assignment description
    assert abs(p.learning_rate - 0.5) < 1e-9, "Default learning_rate should be 0.5"
    # Check predict runs
    yhat = p.predict(np.array([0.5, -0.2]))
    assert yhat in (-1, 1)

def test_perceptron_learns_linearly_separable():
    ns = _exec_notebook(NB_PATH)
    Perceptron = ns["Perceptron"]
    rng = np.random.default_rng(123)
    X = rng.normal(size=(200,2))
    y = np.where(X[:,0] - X[:,1] > 0.0, 1, -1)
    p = Perceptron(input_size=2)
    p.train(X, y, epochs=20)
    preds = np.array([p.predict(x) for x in X])
    acc = (preds == y).mean()
    assert acc > 0.9, f"Perceptron should reach >90% accuracy on a linearly separable dataset, got {acc:.3f}"

def test_perceptron_struggles_on_circles():
    ns = _exec_notebook(NB_PATH)
    Perceptron = ns["Perceptron"]
    try:
        from sklearn.datasets import make_circles
    except Exception as e:
        # If sklearn isn't available in the grading env, skip this test gracefully
        import pytest
        pytest.skip("scikit-learn not available in environment")
    X, y01 = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=0)
    y = np.where(y01==1, 1, -1)
    p = Perceptron(input_size=2)
    p.train(X, y, epochs=50)
    preds = np.array([p.predict(x) for x in X])
    acc = (preds == y).mean()
    assert acc < 0.9, "On concentric circles, a plain perceptron should not exceed 90% accuracy"
