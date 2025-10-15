# Screenshots Directory

This directory should contain the following screenshots required for the project submission:

## Required Screenshots

### 1. `example.png`
**What to capture:** FastAPI interactive documentation showing the Pydantic example
- Navigate to `http://localhost:8000/docs` (or your deployed URL + /docs)
- Click on the POST `/predict` endpoint
- Click "Try it out"
- Show the pre-filled example data from the Pydantic schema
- Capture the entire request body example

**Purpose:** Demonstrates that FastAPI automatically generates documentation with examples from Pydantic models

---

### 2. `continuous_integration.png`
**What to capture:** GitHub Actions workflow successfully passing
- Go to your GitHub repository
- Click on "Actions" tab
- Select the latest successful workflow run
- Show all steps completed with green checkmarks:
  - Checkout code ✓
  - Set up Python ✓
  - Install dependencies ✓
  - Lint with flake8 ✓
  - Run unit tests with pytest ✓

**Purpose:** Proves CI is set up and all tests pass

---

### 3. `continuous_deloyment.png`
**What to capture:** Heroku dashboard showing continuous deployment enabled
- Log in to Heroku Dashboard
- Select your app
- Go to "Deploy" tab
- Show:
  - GitHub repository connected
  - Automatic deploys enabled
  - Latest deployment successful

**Purpose:** Proves continuous deployment is configured

---

### 4. `live_get.png`
**What to capture:** Web browser showing successful GET request to deployed API
- Open browser to your Heroku app URL (e.g., `https://your-app-mlops.herokuapp.com/`)
- Ensure the URL is visible in the address bar
- Show the JSON response from the root endpoint:
  ```json
  {
    "message": "Welcome to the Census Income Prediction API!",
    "description": "Use POST /predict to get salary predictions...",
    "docs": "Visit /docs for interactive API documentation"
  }
  ```

**Purpose:** Proves the API is deployed and accessible

---

### 5. `live_post.png`
**What to capture:** Terminal output from running `scripts/query_live.py`
- Run: `python scripts/query_live.py` with your Heroku URL
- Capture terminal output showing:
  - GET / endpoint test (status 200 + response)
  - POST /predict with high income profile (status 200 + prediction)
  - POST /predict with low income profile (status 200 + prediction)
  - All tests passing summary

**Purpose:** Proves the API accepts POST requests and returns predictions

---

## How to Take Screenshots

### Windows
- **Snipping Tool:** Windows Key + Shift + S
- **Full Screen:** PrtScn key
- **Active Window:** Alt + PrtScn

### Mac
- **Selection:** Cmd + Shift + 4
- **Full Screen:** Cmd + Shift + 3
- **Window:** Cmd + Shift + 4, then press Space

### Linux
- **GNOME:** PrtScn or Shift + PrtScn
- **KDE:** Spectacle (PrtScn)

## Checklist

Before submitting, ensure you have:
- [ ] `example.png` - FastAPI docs with example
- [ ] `continuous_integration.png` - GitHub Actions passing
- [ ] `continuous_deloyment.png` - Heroku CD enabled
- [ ] `live_get.png` - Browser showing GET response
- [ ] `live_post.png` - Terminal showing POST results

All screenshots should be clear, readable, and show the full context needed to verify the requirement.

