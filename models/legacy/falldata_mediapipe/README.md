# Legacy MediaPipe fall models

This directory contains archived RandomForest models from the original
MediaPipe-based falldata package. Production does not load these files.

- Active production auxiliary model:
  `models/experiments/yolo_pose_fall_cam2_continuous_200_80_640.pkl`
- Legacy reproduction settings:
  `config/legacy/falldata-mediapipe.env.example`
- Legacy model runtime requirement:
  scikit-learn 1.3.x and the dedicated `.venv-falldata` environment

The `FNF_RF_SMOTE_CAM_1.pkl` through `FNF_RF_SMOTE_CAM_8.pkl` files are kept
only for historical comparison and manual reproduction.
