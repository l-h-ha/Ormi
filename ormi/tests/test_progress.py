import argparse
import io
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
from tqdm.auto import tqdm
from tqdm import auto as tqdm_auto

import ormi as om

###
###
###

class Terminal(io.StringIO):
    def isatty(self):
        return True

###
###
###

def test_minibatch_progress_preserves_training_and_loss_read_count(api_device, monkeypatch, capsys):
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "minibatch.py"))
    train = example["train"]
    stream = Terminal()
    bars = []

    def progress(*args, **kwargs):
        bar = tqdm(*args, file=stream, mininterval=0, ascii=True, **kwargs)
        bars.append(bar)
        return bar

    reads = []
    item = om.Tensor.item

    def counted_item(self):
        reads.append(self)
        return item(self)

    monkeypatch.setattr(tqdm_auto, "tqdm", progress)
    monkeypatch.setattr(om.Tensor, "item", counted_item)
    enabled = train(epochs=2, device=api_device)
    assert len(reads) == 16
    assert len(bars) == 1 and bars[0].n == 16
    assert all(text in stream.getvalue() for text in ["Epoch 1/2", "Epoch 2/2", "100%", "loss="])
    assert "epoch " not in capsys.readouterr().out

    reads.clear()
    stream.seek(0)
    stream.truncate()
    disabled = train(epochs=2, device=api_device, progress=False)
    assert len(reads) == 16
    assert stream.getvalue() == ""
    assert capsys.readouterr().out.count("epoch ") == 2

    for lhs, rhs in zip(enabled.parameters(), disabled.parameters(), strict=True):
        np.testing.assert_array_equal(lhs.numpy(), rhs.numpy())

def test_pinn_progress_samples_loss_only_at_report_intervals(api_device, monkeypatch, capsys):
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "pinn.py"))
    train = example["train"]
    stream = Terminal()
    bars = []

    def progress(*args, **kwargs):
        bar = tqdm(*args, file=stream, mininterval=0, ascii=True, **kwargs)
        bars.append(bar)
        return bar

    reads = []
    item = om.Tensor.item

    def counted_item(self):
        reads.append(self)
        return item(self)

    monkeypatch.setattr(tqdm_auto, "tqdm", progress)
    monkeypatch.setattr(om.Tensor, "item", counted_item)
    args = argparse.Namespace(
        epochs=5, points=2, width=2, learning_rate=0.01,
        device=api_device, report_every=3, seed=7, no_progress=False,
    )
    enabled = train(args)
    assert len(reads) == 3
    assert bars[0].n == 5
    assert "100%" in stream.getvalue() and "loss=" in stream.getvalue()
    assert capsys.readouterr().out == ""

    reads.clear()
    stream.seek(0)
    stream.truncate()
    args.no_progress = True
    disabled = train(args)
    assert len(reads) == 3
    assert stream.getvalue() == ""
    assert capsys.readouterr().out.count("epoch ") == 3

    for lhs, rhs in zip(enabled.parameters(), disabled.parameters(), strict=True):
        np.testing.assert_array_equal(lhs.numpy(), rhs.numpy())

def test_import_and_graph_construction_do_not_load_tqdm():
    script = """
import importlib.abc
import sys

class NoProgress(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tqdm' or fullname.startswith('tqdm.'):
            raise ModuleNotFoundError('tqdm is not installed')

sys.meta_path.insert(0, NoProgress())
import ormi as om
assert om.Tensor
output = om.tensor([1.0, 2.0]).sin()
assert not output.realized
assert 'tqdm' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, timeout=30)
