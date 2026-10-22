# ML Playground (Python + Streamlit)

An interactive AI/ML project: pick a dataset (Iris, Wine, Breast Cancer), pick a model
(k-NN, Decision Tree, Random Forest, Logistic Regression), tune its settings, view
accuracy / confusion matrix, and make live predictions.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Make it public (free)
1. Create a GitHub repo and upload `app.py` and `requirements.txt`.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", choose the repo, set main file to `app.py`, click Deploy.
4. You get a public link like `https://your-app.streamlit.app`.
