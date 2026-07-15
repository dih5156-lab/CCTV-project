import cv2
import numpy as np

from scripts.datasets.prepare_balanced_color_classification import _recolor_apparel


def test_recolor_apparel_preserves_background_and_changes_garment_hue():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[20:80, 20:80] = (255, 0, 0)  # blue garment square

    recolored = _recolor_apparel(
        image,
        source_color="blue",
        target_color="purple",
        rng=__import__("random").Random(42),
    )

    assert recolored is not None
    hsv = cv2.cvtColor(recolored, cv2.COLOR_BGR2HSV)
    assert 140 <= int(hsv[50, 50, 0]) <= 150
    assert np.all(recolored[0, 0] >= 245)
