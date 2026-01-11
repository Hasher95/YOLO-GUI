# YOLO Training GUI

A simple and comprehensive graphical interface for training and using YOLO (v8 and v11) models. Built with **CustomTkinter** and **Ultralytics**.

## Features

*   **Training**: Configure epochs, batch size, image resolution, and device selection (GPU or CPU).
*   **Visualization**: Real-time display of Loss and precision (mAP) curves during training, with automatic "Best Epoch" markers.
*   **Live Monitor**: Dedicated tab that cycles between real-time detections and updated result charts.
*   **Dataset Builder**: Tool to split images and labels into Train/Val folders and automatically generate the `data.yaml` file.
*   **Advanced Inference**:
    *   Batch processing for images or videos.
    *   **Side-by-Side Comparison**: Run two models simultaneously to visually compare their results.
    *   Real-time Confidence and IoU threshold adjustment via sliders.
*   **Export**: Export trained models to various formats (ONNX, TFLite, etc.).
*   **Persistence**: The application remembers your last used directories and settings.

## Installation and Usage

### Option 1: Using the Executable (Recommended)
If you prefer not to manage Python manually, simply run:
`build_exe.bat`
This script will automatically:
1. Install all necessary dependencies.
2. Create a `dist/` folder containing the standalone application.

### Option 2: Run with Python
1. Clone the repository:
   ```bash
   git clone https://github.com/Hasher95/YOLO-GUI.git
   ```
2. Install libraries:
   ```bash
   pip install customtkinter ultralytics pillow matplotlib pandas
   ```
3. Launch the app:
   ```bash
   python train_yolo_gui.py
   ```

## Useful Parameters

*   **lr0 (Learning Rate)**: Adjusts learning speed (default: 0.01).
*   **Momentum**: Helps optimizer stability (default: 0.937).
*   **Mosaic**: Useful for improving detection in complex scenes when enabled.

## License

This project is available under the MIT License.
