import numpy as np

from scripts.datasets import visualize_yolo_pose_selection


def test_build_contact_sheet_pads_incomplete_row_to_requested_columns():
    frames = [
        np.zeros((10, 20, 3), dtype=np.uint8)
        for _ in range(4)
    ]

    sheet = visualize_yolo_pose_selection._build_contact_sheet(
        frames,
        columns=3,
        tile_height=10,
    )

    assert sheet.shape == (20, 60, 3)
