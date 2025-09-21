import nbformat
import numpy as np
from pathlib import Path

# Pfad zum Notebook
HERE = Path(__file__).resolve().parent       # tests/week01
ROOT = HERE.parents[1]                       # Projekt-Root
NB_PATH = ROOT / "a2_BiologicalNeuron.ipynb"

def _strip_magics(src: str) -> str:
    """Entfernt IPython-/Shell-Magics aus einer Codezelle."""
    s = src.lstrip()
    # Cell-magics: ganze Zelle ignorieren
    if s.startswith("%%"):
        return ""
    # Line-magics / Shell-Aufrufe:
    cleaned = []
    for line in src.splitlines():
        ls = line.lstrip()
        if ls.startswith("%") or ls.startswith("!"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def _exec_notebook(path: Path):
    """Führt die Codezellen des Notebooks in einem Namespace aus und gibt diesen zurück."""
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # nur Codezellen, leerer Inhalt raus
    code_cells = [c.source for c in nb.cells if c.cell_type == "code" and c.source.strip()]
    # Magics entfernen
    cleaned_cells = [_strip_magics(src) for src in code_cells]
    # echte Newlines verwenden!
    src = "\n\n".join(s for s in cleaned_cells if s.strip())

    ns = {}
    exec(compile(src, str(path), "exec"), ns, ns)
    return ns

def test_notebook_executes():
    ns = _exec_notebook(NB_PATH)
    assert isinstance(ns, dict)
    assert "tilting_bucket_model" in ns, "Function tilting_bucket_model must be defined"
    assert "Perceptron" in ns, "Class Perceptron must be defined"

def test_tilting_bucket_model_behavior():
    ns = _exec_notebook(NB_PATH)
    f = ns["tilting_bucket_model"]
    spikes, levels = f(threshold=1.0, leak_rate=0.1, input_current=0.2, time_steps=50)
    assert len(spikes) == 50 and len(levels) == 50
    assert set(np.unique(spikes)).issubset({0, 1})
    assert np.all(np.isfinite(levels))

def test_tilting_bucket_model_stochastic():
    ns = _exec_notebook(NB_PATH)
    f = ns.get("tilting_bucket_model_stochastic")
    assert callable(f)
    np.random.seed(0)
    s1, l1 = f(1.0, 0.1, 0.2, 100, noise_std=0.05)
    np.random.seed(0)
    s2, l2 = f(1.0, 0.1, 0.2, 100, noise_std=0.05)
    np.random.seed(1)
    s3, l3 = f(1.0, 0.1, 0.2, 100, noise_std=0.05)
    assert np.array_equal(np.array(s1), np.array(s2)) and np.allclose(l1, l2)
    assert not np.array_equal(np.array(s1), np.array(s3)) or not np.allclose(l1, l3)

def test_tilting_bucket_model_adaptation():
    ns = _exec_notebook(NB_PATH)
    f = ns.get("tilting_bucket_model_adaptation")
    assert callable(f)
    s, l = f(1.0, 0.1, 0.3, 100, adaptation_rate=0.01)
    assert len(s) == 100 and len(l) == 100
    assert set(np.unique(s)).issubset({0, 1})
    assert np.sum(s) > 0

def test_perceptron_init_and_predict_train():
    ns = _exec_notebook(NB_PATH)
    Perceptron = ns["Perceptron"]
    p = Perceptron(input_size=2)
    assert hasattr(p, "weights") and hasattr(p, "bias") and hasattr(p, "learning_rate")
    assert isinstance(p.bias, (int, float))
    assert p.weights.shape == (2,)
    assert abs(p.learning_rate - 0.5) < 1e-9
    yhat = p.predict(np.array([0.5, -0.2]))
    assert yhat in (-1, 1)

def test_perceptron_learns_linearly_separable():
    ns = _exec_notebook(NB_PATH)
    Perceptron = ns["Perceptron"]
    rng = np.random.default_rng(123)
    X = rng.normal(size=(200, 2))
    y = np.where(X[:, 0] - X[:, 1] > 0.0, 1, -1)
    p = Perceptron(input_size=2)
    p.train(X, y, epochs=20)
    preds = np.array([p.predict(x) for x in X])
    acc = (preds == y).mean()
    assert acc > 0.9, f"Perceptron should reach >90% accuracy; got {acc:.3f}"

def test_perceptron_struggles_on_circles():
    ns = _exec_notebook(NB_PATH)
    Perceptron = ns["Perceptron"]
    try:
        from sklearn.datasets import make_circles
    except Exception:
        import pytest
        pytest.skip("scikit-learn not available in environment")
    X, y01 = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=0)
    y = np.where(y01 == 1, 1, -1)
    p = Perceptron(input_size=2)
    p.train(X, y, epochs=50)
    preds = np.array([p.predict(x) for x in X])
    acc = (preds == y).mean()
    assert acc < 0.9, "Plain perceptron should not exceed 90% on concentric circles"
