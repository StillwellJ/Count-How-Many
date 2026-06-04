# TripCheck Crash Detection

Continuously polls [TripCheck](https://www.tripcheck.com/) traffic cameras (Oregon DOT), runs a trained **Ultralytics YOLO** model on each still image, and routes results by classification: **no crash** vs **severity** (moderate, severe).

Includes a **live web dashboard** that streams each camera frame and shows where the image would be saved.

## Requirements

- Python 3.10+
- A **TripCheck Data API** subscription key ([ODOT API Portal](https://apiportal.odot.state.or.us/product#product=tripcheck-api-data))
- Your trained crash model weights (`.pt`) — see [Model weights](#model-weights)

## Project layout

| File / folder | Purpose |
|---------------|---------|
| `main.py` | CLI pipeline — poll cameras, run inference, save crash images |
| `web_app.py` | Live dashboard — stream inference results to the browser |
| `crash_detector.py` | YOLO inference (classification or detection) with TripCheck frame cropping |
| `tripcheck_fetcher.py` | TripCheck camera inventory and image download |
| `test_detector_local.py` | Test inference on a single local image (no API key) |
| `config.yaml` | Weights path, confidence, severity mapping, crop settings |
| `templates/live.html` | Live dashboard UI |
| `best.pt` | Trained crash classifier (not committed to git — add locally) |
| `output/` | Saved crash images (gitignored) |

## Setup

1. **Clone the repo** and create a virtual environment:

   ```bash
   cd Capstone2026
   python -m venv venv
   venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```

2. **Get a TripCheck API key**

   - Sign up at the [TripCheck Data API](https://apiportal.odot.state.or.us/product#product=tripcheck-api-data).
   - Create a subscription and copy your key.

3. **Configure environment**

   Create a `.env` file in the project root:

   ```env
   TRIPCHECK_API_KEY=your_subscription_key_here
   ```

   Optional override for weights path:

   ```env
   YOLO_WEIGHTS_PATH=best.pt
   ```

4. **Add model weights** — see [Model weights](#model-weights) below.

5. **Review `config.yaml`** — especially `severity_folders` if your model classes differ.

## Model weights

The pipeline expects a trained crash model at `best.pt` (configured in `config.yaml` → `weights_path`).

The current model is a **3-class YOLO classifier**:

| Class | Output folder |
|-------|----------------|
| `moderate-accident` | `output/severity_moderate/` |
| `no-accident` | `output/no_crash/` (not saved by `main.py`) |
| `severe-accident` | `output/severity_severe/` |

**Important:** Do not use `weights/best.pt` unless you replace it. That file may be an unrelated pretrained model. Place your trained weights as `best.pt` in the project root, or set `YOLO_WEIGHTS_PATH` in `.env`.

Weight files (`.pt`) are gitignored because of size. After cloning, copy your weights in manually.

The detector supports both **classification** (`YOLO-cls`) and **object detection** (`YOLO detect`) weights. Classification uses the top predicted class; detection uses bounding boxes.

## TripCheck frame cropping

TripCheck API images include headers and footers outside the camera view. Before inference, the pipeline crops to the road scene using `inference_crop` in `config.yaml` (default: 12% top, 18% bottom). Adjust if your tiles look different, or set `enabled: false` to use the full frame.

## Testing

See [TESTING.md](TESTING.md) for step-by-step details.

Quick checks:

```bash
# Single local image (no API key)
python test_detector_local.py path/to/image.jpg

# One API cycle, 3 cameras
python main.py --test --cameras 3

# One full cycle (all cameras), then exit
python main.py --test
```

## Running the CLI pipeline

```bash
python main.py
```

This will:

- Fetch all TripCheck cameras across Oregon (cached 24 hours).
- Every `poll_interval_seconds` (default 60), download the current still from each camera.
- Crop the frame, run inference, and save **crash** images under `output/severity_*/`.
- Run until you stop it with Ctrl+C.

Limit cameras for testing:

```bash
python main.py --test --cameras 5
```

## Live web dashboard

Stream inference results in the browser — each camera image appears with its classification and destination folder.

```bash
python web_app.py
```

Open **http://127.0.0.1:5000**

Options:

```bash
# All cameras per cycle
python web_app.py --cameras 0

# Quick demo: 5 cameras, 1 second between each
python web_app.py --cameras 5 --delay 1

# Custom port
python web_app.py --port 8080
```

| Flag | Default | Description |
|------|---------|-------------|
| `--cameras N` | 10 | Cameras per cycle; use `0` for all |
| `--delay SEC` | 1.5 | Seconds between each camera in a cycle |
| `--port` | 5000 | Web server port |

## Configuration summary

| Setting | Description |
|---------|-------------|
| `weights_path` | Path to your `.pt` weights (default: `best.pt`) |
| `inference_crop` | Fractional crop applied before inference |
| `severity_folders` | Map class index or name → output folder |
| `confidence_threshold` | Min top-1 confidence to act on a prediction (0–1) |
| `poll_interval_seconds` | Seconds between full camera cycles |
| `output_dir` | Base folder for severity output |
| `TRIPCHECK_API_KEY` | TripCheck API key (`.env`) |
| `YOLO_WEIGHTS_PATH` | Optional override for `weights_path` (`.env`) |

### Confidence threshold

For the classification model, `confidence_threshold` applies to the **top predicted class**. A good starting point is **0.60**. Lower values accept more uncertain predictions; higher values reduce false alarms but may miss weak detections.

If the model is confidently wrong (e.g. 95% on a normal traffic frame), that is a training/data issue — raising the threshold alone will not fix it.

## References

- [TripCheck API](https://www.tripcheck.com/Pages/API)
- [TripCheck API Getting Started Guide (PDF)](https://www.tripcheck.com/pdfs/TripCheckAPI_Getting_Started_GuideV5.pdf)
- Camera inventory returns `cctv-url` for each still; scripts download that URL on each poll cycle.

If your model uses different class names or severity levels, update `severity_folders` in `config.yaml` to match.
