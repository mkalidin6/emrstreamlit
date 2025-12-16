# EMR Assistant (Web App)

This is a web UI wrapper around your original Tkinter EMR assistant logic (unchanged).

## Files
- `emr_core.py`  -> your original ~1500-line code (UNCHANGED)
- `app.py`       -> Streamlit web UI (ONLY UI layer changed)
- `requirements.txt`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
- Streamlit Community Cloud: push these files to GitHub and deploy `app.py`
- Hugging Face Spaces: choose Streamlit template
- Docker/self-host: install requirements and run `streamlit run app.py`

## Notes
- Ensure the MIMIC-IV demo folder path in `emr_core.py` (BASE) is correct for the deployment environment.
