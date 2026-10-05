"""Minimal runner for environments without pytest:  python tests/run_without_pytest.py"""
import importlib, inspect, sys, tempfile, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
fails = 0
for mod in ["test_data_features", "test_pipeline_explain", "test_continual", "test_loader"]:
    m = importlib.import_module(f"tests.{mod}")
    for n, f in inspect.getmembers(m, inspect.isfunction):
        if n.startswith("test_"):
            try:
                f(Path(tempfile.mkdtemp())) if "tmp_path" in inspect.signature(f).parameters else f()
                print("PASS", mod, n)
            except Exception:
                fails += 1; print("FAIL", mod, n); traceback.print_exc()
sys.exit(1 if fails else 0)
