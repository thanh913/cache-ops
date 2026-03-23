import os

import pytest
import torch

from memory_ops import CacheContext
from memory_ops_runtime import GenerationConfig
from memory_ops_runtime.backends import DummyBackend


@pytest.mark.gpu
@pytest.mark.logical
def test_gpu_machine_and_logical_path_smoke() -> None:
    if not torch.cuda.is_available() and os.getenv("FORCE_CUDA", "0") != "1":
        pytest.skip("CUDA not available")

    backend = DummyBackend()
    block = backend.prefill([0, 1, 2, 3, 4])
    out = backend.generate(CacheContext(block), "pick", GenerationConfig(do_sample=False))
    assert out.answer_text == "A"
