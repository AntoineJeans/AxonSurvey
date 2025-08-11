# 🐭 Rat Brain GUI – Implementation Plan

Web app frontend to explore structured rat brain image datasets and interact with existing segmentation + feature map models. Built for non-technical local users.

---

## 🗂️ 1. Project Structure

```
rat_brain_gui/
├── static/              # Static files (CSS, JS, pre-rendered outputs)
├── templates/           # Jinja2 templates for Flask pages
├── app.py               # Flask app + routing
├── config.py            # Constants, directory paths
├── comparisons/         # Output images for comparison page
└── quicktrace_outputs/  # Output directory for traced uploads

data is stored in ../data/
```

---

## 🌐 2. Pages Overview

| Page        | URL                   | Description                             |
| ----------- | --------------------- | --------------------------------------- |
| Index       | `/`                   | Scrollable list of rat IDs              |
| Rat Page    | `/rat/<rat_id>`       | List of regions for the rat             |
| Region      | `/rat/<rat_id>/<reg>` | View image, segmentation, feature map   |
| Comparison  | `/compare`            | Select rats, show comparison graphs     |
| Quick Trace | `/quicktrace`         | Upload any grayscale image and trace it |

---

## 🧠 3. Backend Logic (Flask)

### General

* Serve via Flask, no user auth.
* Load directory structure on startup.
* Expose JSON APIs only if needed for frontend.

### Index Page

* Scan `../data/` for top-level rat folders.
* Pass list of rat IDs to template.
* Basic scroll or dropdown UI.

### Rat Page

* On click from index, route to `/rat/<rat_id>`.
* List subfolders (regions).
* Include optional metadata (placeholder, or read from `meta.json` later).

### Region Page

* Show original image from `/data/<rat_id>/<region>/image.tif` (or similar).
* Check cache:

  * If segmentation exists at `/cache/<rat_id>/<region>_seg.TIF`, load it.
  * Else, run segmentation model and save to cache.
  * Same for feature map.
* Render both outputs side-by-side using `<img>` tags.
* Apply matplotlib colormap to feature map before saving if grayscale.
* Allow re-generation via button (if needed later).

### Comparison Page

* Render list of available rats and predefined regions.
* User selects 2+ rats; selects region from dropdown (fixed options).
* On submit:

  * Load precomputed feature maps from cache.
  * Compute comparisons (e.g. mean per group).
  * Generate matplotlib plots (bar charts, etc).
  * Save to `/comparisons/` and display as static `<img>`.

### Quick Trace Page

* Route: `/quicktrace`
* Form with file upload input for single image.
* On submit:

  * Ensure image is 2D grayscale (reject color/RGB images).
  * Run tracing model: `trace(image) -> traced_image`
  * Save result to `/quicktrace_outputs/` with unique filename.
  * Show result inline with download link.
* No use of cache, region, or rat info — standalone image-to-image tracing.
* No performance guarantees.

---

## 🖼️ 4. Image & Feature Map Handling

### Loading

* PIL or OpenCV for image loading.
* Resize only for display; keep full res for processing.

### Segmentation

* `segment(image) -> mask` — already exists.
* Save as TIF (same shape as input image).
* Use cache based on rat+region key.

### Feature Map

* `extract_features(image) -> feature_map`
* Normalize and colormap (e.g., viridis) using matplotlib or OpenCV LUT.
* Save as TIF for fast reload.

### Tracing (Quick Trace)

* `trace(image) -> traced_output`
* Accepts any grayscale image.
* Result shown inline and saved in `/quicktrace_outputs/`
* No caching or smart handling, just runs and returns.

---

## 🧱 5. Frontend Notes (Basic Flask Template)

* Jinja2 templates, Bootstrap for layout.
* Use tabs or accordion for regions.
* On Region page, two columns: Image | Segmentation + Feature map.
* Color legends optional (can add later).
* Comparison page form: multi-select rats, dropdown region, submit button.
* Graphs saved as static images, not generated client-side.
* Quick Trace page: single file upload, show output and download button.

---

## 🗃️ 6. Metadata Handling

* Optional JSON file at `/data/<rat_id>/meta.json` for future info.
* Sample format:

```json
{
  "id": "RatA",
  "date": "2023-04-02",
  "group": "experimental",
  "notes": "Low signal in region2"
}
```

* Show these in the Rat or Region pages.

---

## 🧪 7. Testing / Utilities

* Add CLI tool to (re)generate all segmentations/feature maps:
  `python tools/precache.py`
* Use Flask debug mode for dev.
* Sample data loader function for dev/testing.
* Add test image(s) to try Quick Trace manually.

---

## 🧰 8. Extras & Future Additions (not now)

* Optional region annotations
* Export comparison results as CSV
* Persistent grouping/saved comparison profiles
* Drag-to-compare or overlay toggle
* Switch to React frontend if complexity grows
* Quick Trace batch upload (multi-image support)

---

## ✅ MVP Completion Checklist

* [ ] Flask app initializes, loads rats from disk
* [ ] Index page shows all rats
* [ ] Rat page shows all regions
* [ ] Region page loads original image
* [ ] Region page shows cached or new segmentation + feature map
* [ ] Comparison page accepts group selection
* [ ] Comparison page renders static bar chart
* [ ] Quick Trace page accepts image, runs tracing, shows + downloads result
* [ ] Simple CSS layout for usability
* [ ] Works locally via `flask run`
