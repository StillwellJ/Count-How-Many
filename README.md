# TripCheck Crash Detection

Continuously polls [TripCheck](https://www.tripcheck.com/) traffic cameras (Oregon DOT), runs a **YOLOv8** crash-detection model using your trained weights, and saves images into folders: **no crash** vs **by severity** (e.g. minor, moderate, severe).

## Requirements

- Python 3.10+
- A **TripCheck Data API** subscription key (free from [ODOT API Portal](https://apiportal.odot.state.or.us/product#product=tripcheck-api-data))
- Your **YOLOv8 weights** (`.pt`) from a crash-detection model

## Setup

1. **Clone / open the project** and create a virtual environment:

   ```bash
   cd "Capstone 2026"
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Get a TripCheck API key**

   - Sign up at [TripCheck Data API](https://apiportal.odot.state.or.us/product#product=tripcheck-api-data).
   - Create a subscription and copy your key.

3. **Configure environment**

   - Copy `.env.example` to `.env`.
   - Set your API key:
     ```env
     TRIPCHECK_API_KEY=your_subscription_key_here
     ```

4. **Add your YOLOv8 weights**

   - Place your `.pt` file in the project (e.g. `weights/crash_detection.pt`).
   - Set the path in `config.yaml` under `weights_path`, or in `.env` as `YOLO_WEIGHTS_PATH`.

5. **Adjust severity mapping** (optional)

   - Edit `config.yaml` → `severity_folders` so class indices (or names) match your model:
     - `0` → `no_crash`
     - `1` → `severity_minor`
     - `2` → `severity_moderate`
     - `3` → `severity_severe`
   - Set `confidence_threshold` and `use_highest_severity` as needed.

## Testing

Before running continuously, you can:

- **Test the detector on one image** (no API key needed):  
  `python test_detector_local.py path/to/image.jpg`
- **Test one cycle with a few cameras**:  
  `python main.py --test --cameras 3`
- **Test one full cycle then exit**:  
  `python main.py --test`

See [TESTING.md](TESTING.md) for step-by-step testing.

## Running

```bash
python main.py
```

The program will:

- Fetch the list of all TripCheck cameras (cached for 24 hours).
- Every **poll_interval_seconds** (default 60), download the current still image from each camera.
- Run YOLOv8 inference with your weights.
- Save each image under `output/`:
  - **no_crash** – no crash detected above the confidence threshold.
  - **severity_*** – one folder per severity level when a crash is detected.

It runs indefinitely; stop with Ctrl+C.

## Configuration summary

| Setting | Description |
|--------|-------------|
| `config.yaml` → `weights_path` | Path to your `.pt` weights file |
| `config.yaml` → `severity_folders` | Map class index/name → folder name |
| `config.yaml` → `confidence_threshold` | Min confidence for a detection (0–1) |
| `config.yaml` → `poll_interval_seconds` | Seconds between full camera cycles |
| `config.yaml` → `output_dir` | Base folder for `no_crash` and severity folders |
| `.env` → `TRIPCHECK_API_KEY` | TripCheck API subscription key |
| `.env` → `YOLO_WEIGHTS_PATH` | Override for `weights_path` |

## If you need more info

- **TripCheck API:** [TripCheck API](https://www.tripcheck.com/Pages/API), [Getting Started Guide](https://www.tripcheck.com/pdfs/TripCheckAPI_Getting_Started_GuideV5.pdf).
- **Camera inventory** returns `cctv-url` for each still image; the script uses that URL to download images on each cycle.

If your model uses different class names or severity levels, update `severity_folders` in `config.yaml` to match.
