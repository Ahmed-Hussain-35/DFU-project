# DFU Analysis Platform

AI-assisted diabetic foot ulcer (DFU) monitoring app built for the E9 internship
problem statement: severity assessment of diabetic foot ulcers from photographs.

The project started as a single-file segmentation demo and has since grown into
a small clinical workflow: patients upload wound photos and get an AI reading,
their assigned doctor reviews the same case with the full image and heuristic
breakdown, and everything is tied to a proper account system instead of a bare
upload form.

## What it does

- Patients register, pick a doctor from the list of registered clinicians, and
  upload photos of a wound.
- Each photo is segmented by a U-Net++ (EfficientNet-B4 encoder) model trained
  on DFUC2022 plus a second wound-segmentation dataset, combined as an ensemble
  so a wound only gets missed if both models miss it.
- The app reports ulcer coverage, a colour/texture-based infection indicator
  (redness, necrosis, pus, swelling, irregularity), and a size-based severity
  label. Wagner grading was deliberately left out — it depends on wound depth
  and exposed bone/tendon, which cannot be judged from a 2D photo, so including
  it would have been more misleading than useful.
- Every visit is saved with a short numeric ID the patient can reference, and
  automatically shows up on their assigned doctor's dashboard — no manual
  code-sharing required.
- Doctors see the same overlay/mask/boundary images and feature breakdown the
  patient saw, can leave notes, confirm or reject the AI-predicted mask, and
  mark a visit as reviewed.
- Patients can add a short note about their symptoms at upload time (e.g. pain
  location, how long it's been there), which is saved against that visit and
  shown to the reviewing doctor.
- Low-quality photos (too dark, too blurry, too small) are flagged before the
  model even runs, so a bad photo doesn't quietly produce a meaningless result.

## Architecture

- **Backend**: FastAPI, SQLAlchemy models, JWT-based auth with patient/doctor
  roles. SQLite by default — swapping to Postgres later is a one-line change
  to `DATABASE_URL` since the code is fully ORM-based.
- **Frontend**: a single React file (loaded via CDN, no build step) served
  directly by FastAPI as a static file.
- **Model**: ONNX Runtime for inference (a PyTorch `.pth` checkpoint also
  works, auto-detected from the file extension). Segmentation output feeds a
  set of hand-written colour/texture heuristics for the infection indicator —
  this part is explicitly a heuristic, not a trained classifier, since no
  infection-labelled dataset was available at the time (DFUC2021/Part-B are
  access-gated). Segmentation itself is a real trained model with measured
  Dice/IoU on held-out data.
- Images (overlay, boundary, mask) and the full feature breakdown are stored
  per visit so a doctor's review loads instantly, without re-running the model.

## Project layout

```
app/
  main.py              FastAPI app, mounts routers and static files
  models.py             SQLAlchemy models (users, patients, doctors, visits, images)
  auth.py                Password hashing, JWT issuing/validation, role checks
  inference.py           Segmentation + infection heuristic pipeline
  routers/
    auth_router.py       Register / login / doctor list
    visits_router.py      Upload, history, doctor review, my-patients
static/
  index.html            Frontend (patient and doctor views)
requirements.txt
```

## Running it

1. Place your model file in the project root, next to `requirements.txt`:
   `dfu_model.onnx` + `dfu_model.onnx.data`, or a `best_model.pth` checkpoint.
2. Install dependencies and run:

```
pip install -r requirements.txt
uvicorn app.main:app --port 8010
```

3. Open `http://localhost:8010`. Register a doctor account first (so patients
   have someone to pick from), then register a patient account.

Set `DFU_MODEL` as an environment variable if your model file has a different
name, e.g. `best_model.pth`.

## Known limitations

- The infection indicator is a rule-based estimate built from wound colour and
  texture, not a model trained on infection-labelled data. It's presented in
  the UI as an "AI visual indicator," not a diagnosis, and should be read that
  way.
- No clinical progression charting yet — visit history exists per patient, but
  trend/healing analysis over time isn't built.
- This is a student project, not a certified medical device. Nothing it
  outputs should be used for an actual clinical decision without a qualified
  clinician in the loop.

## Background

Originally built for the DFUC2022 dataset as a segmentation-only tool, then
extended with an infection heuristic, a second segmentation model trained on
additional wound data (combined via ensemble for better small-wound recall),
and finally restructured into the patient/doctor platform described above.
