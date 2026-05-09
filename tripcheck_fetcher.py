"""
Fetch TripCheck (ODOT) camera inventory and still images.
Uses the TripCheck Data API: https://www.tripcheck.com/Pages/API
Fetches all cameras across Oregon by querying multiple regional bounds and merging.
"""
import logging
import os
import time
from urllib.parse import quote

import requests

# Single-region fallback (Portland); used only when not using all-state
OREGON_BOUNDS = "-122.846803,45.312897,-122.378803,45.568083"

# Regional bounds (minlon, minlat, maxlon, maxlat) covering Oregon. API returns 400 for
# one statewide box, so we query by region and merge. Based on ODOT/TripCheck coverage.
OREGON_REGIONS = [
    "-122.846803,45.312897,-122.378803,45.568083",   # Portland
    "-123.153519,44.819314,-122.649742,45.284286",  # Salem / surrounding
    "-123.235,43.946472,-122.792656,44.226514",     # Eugene
    "-123.8907,42.7235,-122.0779,43.8161",         # Roseburg
    "-123.006094,42.179417,-122.776114,42.508094",  # Medford / Central Point
    "-122.819803,42.009244,-122.520011,42.320272",  # Ashland / Siskiyou
    "-122.553828,42.140986,-121.602086,42.457756",  # Klamath Falls
    "-124.5,43.0,-123.4,46.2",                      # Coast (Astoria to Brookings)
    "-118.0,44.0,-116.5,46.0",                      # Eastern Oregon
]
OREGON_REGION_NAMES = [
    "Portland",
    "Salem / surrounding",
    "Eugene",
    "Roseburg",
    "Medford / Central Point",
    "Ashland / Siskiyou",
    "Klamath Falls",
    "Coast",
    "Eastern Oregon",
]
CCTV_INVENTORY_URL = "https://api.odot.state.or.us/tripcheck/Cctv/Inventory"


def get_camera_inventory(api_key: str, bounds: str | None = None) -> list[dict]:
    """
    Fetch CCTV camera inventory from TripCheck API.
    Returns list of camera dicts with keys: device-id, device-name, cctv-url, etc.
    """
    bounds = bounds or OREGON_BOUNDS
    url = f"{CCTV_INVENTORY_URL}?Bounds={quote(bounds)}"
    headers = {
        "Cache-Control": "no-cache",
        "Ocp-Apim-Subscription-Key": api_key,
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    cameras = data.get("CCTVInventoryRequest") or data.get("CCTVInventoryItem") or []
    if isinstance(cameras, dict):
        cameras = [cameras]
    # Normalize http -> https for cctv-url
    for cam in cameras:
        url_val = cam.get("cctv-url") or cam.get("cctv_url") or ""
        if url_val.startswith("http://"):
            cam["cctv-url"] = url_val.replace("http:", "https:", 1)
        else:
            cam["cctv-url"] = url_val
    return cameras


def get_all_state_cameras(api_key: str) -> list[dict]:
    """
    Fetch cameras from all regions across Oregon, merge and deduplicate by device-id.
    Logs per-region success/failure and total count so you can verify we're pinging all cameras.
    """
    log = logging.getLogger(__name__)
    seen_ids: set[int | str] = set()
    merged: list[dict] = []
    failed_regions: list[str] = []
    for i, bounds in enumerate(OREGON_REGIONS):
        name = OREGON_REGION_NAMES[i] if i < len(OREGON_REGION_NAMES) else f"Region {i}"
        try:
            cameras = get_camera_inventory(api_key, bounds)
        except requests.RequestException as e:
            log.warning("Camera inventory failed for %s: %s", name, e)
            failed_regions.append(name)
            continue
        new_count = 0
        for cam in cameras:
            dev_id = cam.get("device-id") or cam.get("device_id")
            if dev_id is None:
                merged.append(cam)
                new_count += 1
                continue
            if dev_id not in seen_ids:
                seen_ids.add(dev_id)
                merged.append(cam)
                new_count += 1
        log.info("Cameras from %s: %d (total unique so far: %d)", name, new_count, len(merged))
    log.info(
        "Camera inventory complete: %d total cameras from %d regions; %d region(s) failed: %s",
        len(merged),
        len(OREGON_REGIONS) - len(failed_regions),
        len(failed_regions),
        failed_regions if failed_regions else "none",
    )
    return merged


def fetch_camera_image(cctv_url: str, timeout: int = 15) -> bytes | None:
    """
    Download still image from a TripCheck camera URL.
    Returns image bytes or None on failure.
    """
    try:
        resp = requests.get(cctv_url, timeout=timeout, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and not cctv_url.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Some cameras return octet-stream; still try to use it
            pass
        return resp.content
    except requests.RequestException:
        return None


def get_camera_inventory_cached(
    api_key: str,
    bounds: str | None = None,
    cache_ttl_seconds: int = 86400,
    cache_file: str = ".camera_inventory_cache.json",
    all_state: bool = True,
) -> list[dict]:
    """
    Get camera inventory with file-based cache.
    If all_state is True (default), fetches from all OREGON_REGIONS and merges;
    otherwise uses the single bounds parameter.
    """
    import json

    now = time.time()
    if os.path.isfile(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if now - cached.get("_ts", 0) < cache_ttl_seconds:
                return cached.get("cameras", [])
        except (json.JSONDecodeError, KeyError):
            pass
    if all_state:
        cameras = get_all_state_cameras(api_key)
    else:
        cameras = get_camera_inventory(api_key, bounds)
    try:
        with open(cache_file, "w") as f:
            json.dump({"_ts": now, "cameras": cameras}, f, indent=0)
    except OSError:
        pass
    return cameras
