# Intelligent Network Traffic Mining & Cyber Threat Prediction System

## Run in Visual Studio Code

1. Install Python 3.10 or newer.
2. Extract this folder and open it in VS Code.
3. Open Terminal > New Terminal.
4. (Recommended) Create a virtual environment:

   Windows:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

5. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

6. Start the Streamlit application:
   ```powershell
   streamlit run app.py
   ```

7. Streamlit should open the app in your browser. If it does not, open the local URL printed in the terminal (usually http://localhost:8501).

## Features

- Sample synthetic network-traffic dataset, or upload a CSV.
- Dataset preview and summary statistics.
- K-Means clustering.
- Decision Tree attack-type classification and prediction.
- Linear Regression threat-score estimation.
- Isolation Forest anomaly detection and CSV export.

## Important note

The built-in sample data is synthetic and intended for demonstration. Predictions are not a substitute for monitoring and validating real network traffic. Uploaded CSVs should contain useful numeric traffic features; a `label` column is helpful for attack-type classification.
