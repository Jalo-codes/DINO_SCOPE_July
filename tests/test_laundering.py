import os
import tempfile
import pytest

pytest.importorskip('torch')

from PIL import Image as PILImage
from pathlib import Path

from lab_utils.eval.numbers import _load_and_launder


class MockItem:
    def __init__(self, image_path, source="tgif2"):
        self.image = Path(image_path)
        self.source = source


def test_bicubic_laundering():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test image
        img_path = os.path.join(tmpdir, "test_img.png")
        img = PILImage.new("RGB", (100, 100), color="red")
        img.save(img_path)

        item = MockItem(img_path)

        # 1. Test launder_mode='none'
        img_down, img_up = _load_and_launder(item, "none", None)
        assert img_down.size == (100, 100)
        assert img_up.size == (100, 100)

        # 2. Test launder_mode='bicubic_x2'
        img_down, img_up = _load_and_launder(item, "bicubic_x2", None)
        assert img_down.size == (100, 100)
        assert img_up.size == (200, 200)

        # 3. Test launder_mode='bicubic_x4'
        img_down, img_up = _load_and_launder(item, "bicubic_x4", None)
        assert img_down.size == (100, 100)
        assert img_up.size == (400, 400)


def test_real_esrgan_laundering():
    from lab_utils.eval.numbers import _load_prelaundered_image
    class DummyArgs:
        def __init__(self, tgif2_root):
            self.tgif2_root = tgif2_root

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_root = Path(tmpdir) / "tgif2_flux"
        upscaled_root = Path(tmpdir) / "tgif2_flux_esrgan_x2"
        orig_root.mkdir()
        upscaled_root.mkdir()

        # Create original image
        orig_path = orig_root / "test_img.png"
        img = PILImage.new("RGB", (100, 100), color="blue")
        img.save(orig_path)

        # Create upscaled image
        upscaled_path = upscaled_root / "test_img.png"
        img_up = PILImage.new("RGB", (200, 200), color="green")
        img_up.save(upscaled_path)

        item = MockItem(orig_path, source="tgif2")
        args = DummyArgs(str(orig_root))

        # 1. Test folder swap (without explicit prelaundered_root)
        loaded_img = _load_prelaundered_image(item, "real_esrgan_x2", None, img, args=args)
        assert loaded_img.size == (200, 200)

        # 2. Test explicit prelaundered_root
        custom_root = Path(tmpdir) / "custom_sr"
        custom_root.mkdir()
        custom_path = custom_root / "test_img.png"
        img_custom = PILImage.new("RGB", (300, 300), color="red")
        img_custom.save(custom_path)

        loaded_img = _load_prelaundered_image(item, "real_esrgan_x2", str(custom_root), img, args=args)
        assert loaded_img.size == (300, 300)
