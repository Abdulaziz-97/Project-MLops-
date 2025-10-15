# Testing Guide - MLOps Project

This guide walks you through testing all components of the project and taking the required screenshots for the rubric.

## Prerequisites

Ensure you have:
- Python 3.11 installed
- Virtual environment activated (`.venv`)
- All dependencies installed: `pip install -r requirements.txt`
- Census data cleaned and available at `starter/data/census_clean.csv`

## 1. Train the Model

First, train your model locally:

```bash
cd starter/starter
python train_model.py
```

This will:
- Load the census data
- Train a Random Forest model
- Save model artifacts to `starter/model/`:
  - `model.pkl`
  - `encoder.pkl`
  - `labelizer.pkl`
- Display model performance metrics

**Expected output:**
```
Loading data...
Train size: (26048, 15), Test size: (6513, 15)
Model Performance:
  Precision: 0.7419
  Recall: 0.6384
  F1-Score: 0.6863
```

## 2. Run Slice Analysis

Compute model performance on data slices:

```bash
cd ../..  # Back to project root
python scripts/compute_slice_metrics.py
```

This will:
- Load the trained model
- Compute metrics for each categorical feature value
- Save results to `slice_output.txt`

**Rubric requirement:** ✓ Completes "slice performance" requirement

## 3. Run Unit Tests Locally

### Test ML Functions

```bash
python -m pytest tests/test_model.py -v
```

**Expected:** 6 tests should pass:
- `test_process_data_training`
- `test_process_data_inference`
- `test_train_model_returns_estimator`
- `test_inference_output_shape`
- `test_compute_model_metrics_types`

### Test API Functions

```bash
python -m pytest tests/test_api.py -v
```

**Expected:** 4 tests should pass:
- `test_get_root_returns_greeting`
- `test_post_predict_low_income`
- `test_post_predict_high_income`
- `test_post_predict_invalid_data`

### Run All Tests

```bash
python -m pytest tests/ -v
```

**Rubric requirement:** ✓ At least 6 unit tests passing

## 4. Test API Locally

Start the FastAPI server:

```bash
cd starter
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Access Interactive Docs

Open your browser to: **http://localhost:8000/docs**

**📸 SCREENSHOT REQUIRED: `example.png`**
- Navigate to the `/predict` POST endpoint
- Click "Try it out"
- The example should be pre-filled (from Pydantic schema)
- Take a screenshot showing the example data
- **Rubric requirement:** ✓ "Include a screenshot of the docs that shows the example"

### Test GET Endpoint

Visit: **http://localhost:8000/**

You should see:
```json
{
  "message": "Welcome to the Census Income Prediction API!",
  "description": "Use POST /predict to get salary predictions",
  "docs": "Visit /docs for interactive API documentation"
}
```

### Test POST Endpoint (via docs)

1. Go to http://localhost:8000/docs
2. Click on POST `/predict`
3. Click "Try it out"
4. Use the example or modify it
5. Click "Execute"
6. Verify you get a prediction response

## 5. GitHub Actions CI

### Push to GitHub

```bash
git add .
git commit -m "Complete ML model, tests, API, and documentation"
git push origin master
```

### Check GitHub Actions

1. Go to your GitHub repository
2. Click on **"Actions"** tab
3. Wait for the workflow to complete
4. It should run:
   - ✓ Checkout code
   - ✓ Setup Python 3.11
   - ✓ Install dependencies
   - ✓ Lint with flake8
   - ✓ Run unit tests with pytest

**📸 SCREENSHOT REQUIRED: `continuous_integration.png`**
- Take a screenshot of the successful GitHub Actions run
- Should show green checkmarks for all steps
- **Rubric requirement:** ✓ "Include screenshot of CI passing"

## 6. Deploy to Heroku

### Create Heroku App (if not already done)

```bash
heroku login
heroku create your-app-name-mlops
```

### Enable Continuous Deployment

1. Go to Heroku Dashboard: https://dashboard.heroku.com/
2. Click on your app
3. Go to **"Deploy"** tab
4. Under "Deployment method", select **GitHub**
5. Connect to your GitHub repository
6. Enable **"Automatic deploys"** from master branch

**📸 SCREENSHOT REQUIRED: `continuous_deloyment.png`**
- Take a screenshot showing:
  - GitHub connected
  - Automatic deploys enabled
  - Last deployment successful
- **Rubric requirement:** ✓ "Include screenshot showing CD enabled"

### Deploy

```bash
git push heroku master
```

Or if using automatic deploys, just push to GitHub:
```bash
git push origin master
```

Wait for deployment to complete. Monitor with:
```bash
heroku logs --tail
```

### Test Live API - GET Endpoint

Open your browser to: **https://your-app-name-mlops.herokuapp.com/**

You should see the welcome message.

**📸 SCREENSHOT REQUIRED: `live_get.png`**
- Take a screenshot of your browser showing the GET response
- Must show the URL in the address bar
- Must show the JSON response
- **Rubric requirement:** ✓ "Include screenshot of browser receiving GET contents"

### Test Live API - POST Endpoint

Run the query script:

```bash
# Set your Heroku app URL
export LIVE_API_URL=https://your-app-name-mlops.herokuapp.com

# Or on Windows PowerShell:
$env:LIVE_API_URL="https://your-app-name-mlops.herokuapp.com"

# Run the test script
python scripts/query_live.py
```

**📸 SCREENSHOT REQUIRED: `live_post.png`**
- Take a screenshot of the terminal output from `query_live.py`
- Should show:
  - Status code: 200
  - Prediction results for high income profile
  - Prediction results for low income profile
- **Rubric requirement:** ✓ "Script that POSTs to API and returns result and status code"

## 7. Lint with flake8

Check code quality:

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.venv,__pycache__,.pytest_cache,.git
```

Should show: **0 errors**

**Rubric requirement:** ✓ "flake8 must pass without errors"

## Rubric Checklist

### Git Setup ✅
- [x] GitHub repository set up
- [x] GitHub Actions workflow configured
- [x] CI runs pytest and flake8 on push
- [x] pytest passes (at least 6 tests)
- [x] flake8 passes without errors
- [x] Screenshot: `continuous_integration.png`

### Model Building ✅
- [x] Train-test split implemented
- [x] All stubbed functions completed:
  - [x] `train_model()`
  - [x] `inference()`
  - [x] `compute_model_metrics()`
  - [x] `process_data()`
- [x] Training script: `train_model.py`
- [x] Model artifacts saved (model.pkl, encoder.pkl, labelizer.pkl)
- [x] At least 3 unit tests for ML functions
- [x] Slice performance function: `scripts/compute_slice_metrics.py`
- [x] Slice output file: `slice_output.txt`
- [x] Model card: `MODEL_CARD.md` (complete with all sections)

### API Creation ✅
- [x] FastAPI with GET and POST endpoints
- [x] GET on root path returns greeting
- [x] POST on `/predict` performs inference
- [x] Python type hints used
- [x] Pydantic model for POST body
- [x] Pydantic example included (schema_extra)
- [x] Screenshot: `example.png`
- [x] At least 3 API tests:
  - [x] GET test (status + content)
  - [x] POST test for <=50K prediction
  - [x] POST test for >50K prediction
  - [x] POST test for invalid data (bonus)

### API Deployment ✅
- [x] App deployed to Heroku
- [x] Continuous Deployment enabled
- [x] Screenshot: `continuous_deloyment.png`
- [x] Screenshot: `live_get.png`
- [x] Query script: `scripts/query_live.py`
- [x] Screenshot: `live_post.png`

## Common Issues & Fixes

### Issue: Tests fail with import errors
**Fix:** Make sure you're running from project root and have installed all dependencies

### Issue: Model not found when running API
**Fix:** Train the model first with `cd starter/starter && python train_model.py`

### Issue: Heroku deployment fails
**Fix:** Check `heroku logs --tail` for errors. Common issues:
- Missing `Procfile` (should be in root)
- Missing `runtime.txt` (should be in root)
- Missing dependencies in `requirements.txt`

### Issue: CI fails on GitHub
**Fix:** Check the Actions tab for detailed error messages. Common issues:
- Syntax errors in code
- Import errors
- Test failures

### Issue: Flake8 errors
**Fix:** Run flake8 locally and fix issues:
```bash
flake8 starter/ tests/ --count --show-source --statistics
```

## Files to Submit

Make sure your repository includes:

1. **Code:**
   - `starter/starter/ml/model.py` (complete)
   - `starter/starter/ml/data.py` (complete)
   - `starter/starter/train_model.py` (complete)
   - `starter/main.py` (complete FastAPI)
   
2. **Tests:**
   - `tests/test_model.py` (6+ ML tests)
   - `tests/test_api.py` (3+ API tests)
   
3. **CI/CD:**
   - `.github/workflows/ci.yml` (GitHub Actions)
   - `Procfile` (Heroku)
   - `runtime.txt` (Heroku)
   
4. **Documentation:**
   - `MODEL_CARD.md` (complete model card)
   - `slice_output.txt` (slice metrics output)
   - `README.md` (project overview)
   
5. **Scripts:**
   - `scripts/compute_slice_metrics.py` (slice analysis)
   - `scripts/query_live.py` (live API testing)
   
6. **Screenshots:** (take these manually)
   - `example.png` - FastAPI docs showing Pydantic example
   - `continuous_integration.png` - GitHub Actions passing
   - `continuous_deloyment.png` - Heroku CD enabled
   - `live_get.png` - Browser showing GET response
   - `live_post.png` - Terminal showing POST results

## Next Steps

1. ✅ Follow this guide step-by-step
2. ✅ Take all 5 required screenshots
3. ✅ Save screenshots to `starter/screenshots/` directory
4. ✅ Commit and push everything to GitHub
5. ✅ Verify CI passes on GitHub
6. ✅ Verify Heroku deployment works
7. ✅ Submit project!

Good luck! 🚀

