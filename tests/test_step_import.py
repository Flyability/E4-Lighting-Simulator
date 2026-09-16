import os

import numpy as np
import pytest

from lighting_simulator.scene.step_import import STEP_MM_TO_CM, is_step_file, load_step_mesh


def test_is_step_file():
    assert is_step_file("frame.STEP") and is_step_file("a/b.stp") and not is_step_file("frame.stl")


def test_step_roundtrip_keeps_mm_and_origin(tmp_path):
    cq = pytest.importorskip("cadquery")
    path = os.path.join(tmp_path, "box.step")
    # 120 x 80 x 30 mm box with its min corner at (100, 0, 0) mm
    cq.exporters.export(cq.Workplane("XY").box(120, 80, 30, centered=False).translate((100, 0, 0)), path)
    mesh = load_step_mesh(path)
    assert len(mesh.faces) >= 12
    np.testing.assert_allclose(mesh.bounds, [[100, 0, 0], [220, 80, 30]], atol=1e-3)
    assert np.allclose((mesh.bounds[1] - mesh.bounds[0]) * STEP_MM_TO_CM, [12, 8, 3])
