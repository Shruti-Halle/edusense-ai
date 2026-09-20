# EduSense AI · Streamlit edition

EduSense AI is a local, privacy-first early-warning and study coach for teachers,
tutors, and parents. It uses a realistic synthetic dataset when no file is uploaded,
then runs exploratory data analysis and two explainable scikit-learn models.

## Run it

```bash
streamlit run app.py --server.port 5000 --server.address 0.0.0.0
```

The app works without an API key or an upload. To use the UCI Student Performance
file, upload `student-mat.csv` from the sidebar; the delimiter is detected
automatically.

## Pages

- **Home** — class snapshot and the three-step workflow.
- **Explore data** — EDA charts, raw data, and plain-English insights.
- **Check a student** — logistic regression and random forest risk scoring.
- **Study plan** — transparent rule-based plan placeholder.
- **Fair & safe AI** — grouped accuracy and recall checks.

Sex is excluded from the model features and is included only for fairness review.
No student names are requested, and uploads are only held in the current session.