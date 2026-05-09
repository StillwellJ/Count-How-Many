# Testing the TripCheck Crash Detection Pipeline

Use these steps to verify the implementation without running the full loop.

---

## 1. Test the detector only (no API, no TripCheck)

Verifies that your weights load and that `config.yaml` severity mapping is correct.

1. Put any image file in the project (e.g. a screenshot, or a sample from a TripCheck camera).
2. Run:

   ```bash
   python test_detector_local.py path/to/your/image.jpg
   ```

3. Check the output: it should print **Result folder:** `no_crash` or a severity folder (e.g. `severity_minor`).  
   - If you get an error about weights or shape, fix `weights_path` in `config.yaml` or install dependencies.  
   - If the folder is wrong for your model, adjust `severity_folders` in `config.yaml`.

---

## 2. Test one cycle with TripCheck (API + YOLO)

Requires a valid `TRIPCHECK_API_KEY` in `.env`.

1. Ensure `.env` has `TRIPCHECK_API_KEY` and `weights/best.pt` exists.
2. Run a **single cycle** on **3 cameras** only:

   ```bash
   python main.py --test --cameras 3
   ```

3. You should see:
   - Camera inventory fetched (or loaded from cache).
   - “Cycle 1: Processing … cameras (first 3).”
   - For each of the 3 cameras: image downloaded, inference run, image saved under `output/no_crash/` or a severity folder.
   - “Done: N images saved, M crash detections” and “Test complete. Exiting.”

4. Inspect `output/`:
   - `output/no_crash/` should contain the 3 images (unless the model flagged a crash).
   - If any crash was detected, you’ll see files in the corresponding `output/severity_*/` folder.

If you see “Failed to fetch camera inventory” or 401/403, check your API key and TripCheck subscription.

---

## 3. Test one full cycle (all cameras)

Same as step 2 but without limiting cameras (takes longer):

```bash
python main.py --test
```

Use this to confirm the pipeline works for the full camera list before leaving it running.

---

## 4. Run continuously

When tests look good:

```bash
python main.py
```

Stops with Ctrl+C. Images are written every cycle to `output/no_crash/` and `output/severity_*/` as configured.

---

## Quick reference

| Goal                         | Command                                  |
|-----------------------------|------------------------------------------|
| Test detector on one image | `python test_detector_local.py image.jpg`|
| Test pipeline, 3 cameras    | `python main.py --test --cameras 3`      |
| Test pipeline, one cycle    | `python main.py --test`                  |
| Run forever                 | `python main.py`                         |
