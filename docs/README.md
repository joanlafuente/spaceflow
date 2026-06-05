# SpaceFlow User Study

Static GitHub Pages user study for evaluating local appearance control
in 3D asset generation.

---

## Quick start (local testing)

```bash
# from the repo root:
python -m http.server 8000
# open http://localhost:8000 in your browser
```

The placeholder `trials.json` uses random images from picsum.photos,
so the full flow (username → trials → submit) works without any real data.

---

## Setup checklist

### Step 1 — GitHub repo & Pages
1. Create a new GitHub repo (e.g. `spaceflow-study`)
2. Push all files in this folder to the repo
3. Go to repo **Settings → Pages → Source → Deploy from branch → main / (root)**
4. After ~30 seconds, your study is live at `https://USERNAME.github.io/spaceflow-study/`

### Step 2 — Google Apps Script (response collection)
1. Go to [script.google.com](https://script.google.com) and create a new project
2. Paste the code from `apps_script.js` (see below)
3. Replace `YOUR_SHEET_ID` with your Google Sheet's ID (from its URL)
4. Click **Deploy → New deployment → Web app**
   - Execute as: **Me**
   - Who has access: **Anyone**
5. Copy the deployment URL
6. In `static/js/main.js`, replace `"YOUR_APPS_SCRIPT_URL_HERE"` with that URL
7. Push the change to GitHub

### Google Apps Script code to paste:
```javascript
function doPost(e) {
  const sheet = SpreadsheetApp.openById('YOUR_SHEET_ID').getActiveSheet();
  const data = JSON.parse(e.postData.contents);
  sheet.appendRow([
    new Date(),
    data.username,
    data.study_id,
    data.scene_id,
    data.model_a_name,
    data.model_b_name,
    data.answers.q1,
    data.answers.q2,
    data.answers.q3,
    data.timestamp,
  ]);
  return ContentService
    .createTextOutput(JSON.stringify({ success: true }))
    .setMimeType(ContentService.MimeType.JSON);
}
```

### Step 3 — Add your renders
Structure your data folder like this:
```
static/data/
└── chair_01/
    ├── metadata.json        ← prompt + part labels (see example)
    ├── reference_a.webp     ← reference image for part A (e.g. velvet texture)
    ├── reference_b.webp     ← reference image for part B (e.g. oak wood)
    ├── ours/
    │   └── output.webp      ← your method's render (front+back stitched)
    ├── sc/
    │   └── output.webp      ← SpaceControl baseline render
    └── sf_global/
        └── output.webp      ← SpaceFlow global-reference baseline render
```

Then run:
```bash
python generate_trials.py
```

Commit and push the updated `trials.json`.

### Step 4 — Collect responses
Responses land in your Google Sheet automatically.
After the study, export as CSV and run `analyze_results.py` to
generate the bar chart figures for the paper.

---

## Customising

**Number of trials per participant:** edit `TRIALS_PER_PARTICIPANT` in `static/js/main.js`

**Questions:** edit the question text in `index.html` (search for `Q1`, `Q2`, `Q3`)

**Study name:** edit `"SpaceFlow — Local Appearance Control"` in `index.html`

**Adding a second study (τ control):** duplicate `index.html` as `tau_study.html`,
change the `STUDY_ID` constant in a second copy of `main.js`, and add a simple
landing page linking to both.
