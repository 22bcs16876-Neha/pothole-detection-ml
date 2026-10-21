# Pothole Detection — corrected starter project

This is a **binary image-classification** prototype using TensorFlow/Keras,
MobileNetV2 transfer learning, and Streamlit. It classifies the whole image;
it does not draw bounding boxes or locate the pothole.

## Important dataset issue found in the uploaded ZIP

The uploaded ZIP contains 302 images in `dataset/no_pothole/`, but **no pothole
training images** in `dataset/pothole/`. A reliable binary classifier cannot be
trained without examples of both classes. The corrected `train.py` now checks
this and stops with a clear message instead of producing a misleading model.

The ZIP you received already includes 302 no-pothole images. To download public pothole images into the positive class automatically, run this from the project folder (internet required):

```powershell
python download_dataset.py
```

The downloader uses the dataset release published by [jaygala24/pothole-detection](https://github.com/jaygala24/pothole-detection) and copies its image files into `dataset/pothole/`. Review the images and source license/attribution before public redistribution.

Add correctly labeled images to both folders:

```text
dataset/
├── no_pothole/   # normal roads without potholes
└── pothole/      # roads with visible potholes
```

Try to collect a few hundred varied images per class. Avoid duplicates and
make sure every image is in the correct folder. Keep a separate test set if
you want an honest estimate of real-world performance.

## Python and installation

Use the Python version supported by the TensorFlow build you install.
For Python 3.14, `requirements.txt` uses `tf-nightly`, which is a nightly build
and may change. If installation fails, use a supported stable Python version
in a fresh virtual environment.

Windows PowerShell, from this project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train

After downloading or manually adding pothole images:

```powershell
python train.py
```

The model is saved to `models/pothole_model.keras`. Training uses the fixed
class order `no_pothole = 0`, `pothole = 1`.

## Run the app

```powershell
python -m streamlit run app.py
```

Open the Local URL printed in the terminal, usually `http://localhost:8501`.

## Fix applied

The model built by `train.py` includes MobileNetV2 `preprocess_input` inside
the model graph. The earlier app also applied `preprocess_input` before calling
the model, causing **double preprocessing**. The corrected `app.py` passes
raw 0–255 RGB pixels, matching the training pipeline.

## Limitations

- The app uses a 0.5 threshold, not a calibrated or validated operating point.
- A displayed model score is not a guarantee of correctness.
- This is not an object detector and cannot draw pothole boxes.
- Do not use it for road-safety or municipal decisions without rigorous testing.
