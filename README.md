# AirAware — Streamlit App

This is the AirAware Smart Air Quality Forecasting Streamlit app.
Files included:
- app.py
- utils.py
- models.py
- aqi.py
- requirements.txt

## Run locally

1. Create virtualenv and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. Run:
   ```
   streamlit run app.py
   ```

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io and connect your GitHub repo.
3. Choose the main file `app.py` and deploy.

Note: Some packages (prophet, tensorflow, pmdarima) are heavy; Streamlit Cloud may require additional build configuration.
