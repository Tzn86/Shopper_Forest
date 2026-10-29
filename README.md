# Will this visitor buy?

A small, complete machine learning project: a **Random Forest** that predicts whether a visit to an online shop ends in a purchase, with an API and a web app.

Dataset: [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (UCI, 12,330 website visits, CC BY 4.0).

## Project files

| File | What it does |
|---|---|
| `train.py` | Step 1 + 2: explores the data, trains the Random Forest, prints how good it is, saves the model |
| `api.py` | Step 3: a FastAPI server with `/api/meta` and `/api/predict`, and it serves the web app |
| `frontend/index.html` | Step 4: the web app (plain HTML, CSS and JavaScript, no build step) |
| `model/` | Created by `train.py`: the saved model and `meta.json` |
| `reports/` | Created by `train.py`: a feature importance chart |

## How to run it

You need Python 3.10 or newer.

**1. Create a virtual environment and install packages**

```bash
cd shopper-forest
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Train the model** (downloads the dataset the first time)

```bash
python train.py
```

Read the output carefully: it shows the data exploration, the test results, and the most important features.
If the download fails, download `online_shoppers_intention.csv` from the UCI link above and put it in a `data/` folder.

**3. Start the API**

```bash
uvicorn api:app --reload
```

**4. Open the web app**

Go to <http://localhost:8000>. Move the sliders or try a preset visitor, and watch the 300 trees vote.

You can also explore and test the API directly at <http://localhost:8000/docs> (FastAPI creates this page automatically).

## Things to notice (what you'll learn)

- **Accuracy can lie.** Only about 15% of visits end in a purchase, so a model that always says "won't buy" is about 85% accurate and completely useless. That's why `train.py` also shows **precision** and **recall**, and why the forest uses `class_weight="balanced"`.
- **Probability vs. votes.** The percentage comes from `predict_proba`, which averages every tree's confidence. The vote count is each tree's yes/no answer. They're close, but not always identical.
- **Feature importance is a UX insight.** It shows which visitor behaviors the model relies on most, which can suggest what to improve or test on a real shop.
- **The app sends only some fields.** Fields without a slider (browser, region, traffic source...) use typical values from the training data. See `build_row()` in `api.py`.

## Ideas to extend it

1. Add a threshold slider: only flag visitors as "likely buyers" above 70%, and see how precision and recall change.
2. Compare with gradient boosting (`HistGradientBoostingClassifier`) in `train.py`.
3. Rebuild the frontend in React, using the same two API endpoints.
4. Show a per-visitor explanation using the `shap` library.
