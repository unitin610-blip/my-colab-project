# ShopSafe AI

A Streamlit frontend for a machine-learning model that classifies URLs as potentially **phishing** or **legitimate**.

The app takes a URL, extracts the same URL-level features used by the trained model, and displays the predicted class, phishing probability, legitimate probability, and risk level.

## Project structure

```text
my-colab-project/
├── app.py
├── requirements.txt
├── url_deploy_model.pkl
├── url_deploy_features.pkl
├── Untitled5.ipynb
└── README.md
```

## What the app does

1. Accepts a website URL.
2. Extracts 17 URL-based features.
3. Loads the saved Random Forest model.
4. Predicts:
   - Phishing
   - Legitimate
5. Displays:
   - Risk level
   - Phishing probability
   - Legitimate probability

The app does **not** open or scrape the submitted website. It analyzes the URL itself.

---

# Run locally

## Prerequisites

Install **Python 3.10+**.

Check your installation:

### Windows

```powershell
py --version
```

### macOS / Linux

```bash
python3 --version
```

---

## Windows setup

Open **PowerShell** and clone the repository:

```powershell
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
```

Create a virtual environment:

```powershell
py -3 -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the app:

```powershell
streamlit run app.py
```

Streamlit will show a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

### If PowerShell blocks activation

You can either run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

and then:

```powershell
.venv\Scripts\Activate.ps1
```

or use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

Then install dependencies and run:

```cmd
pip install -r requirements.txt
streamlit run app.py
```

---

# macOS / Linux setup

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# Stopping the app

In the terminal running Streamlit, press:

```text
Ctrl + C
```

---

# Updating the project

After changing code:

```bash
git add .
git commit -m "Update Streamlit frontend"
git push
```

---

# Important files

### `app.py`

The Streamlit frontend and URL feature-extraction/inference logic.

### `url_deploy_model.pkl`

The trained machine-learning model.

### `url_deploy_features.pkl`

The saved feature order used when running the model.

### `requirements.txt`

Python dependencies required to run the application.

### `Untitled5.ipynb`

The original notebook containing the model-development workflow.

---
