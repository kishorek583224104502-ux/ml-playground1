"""ML Playground: an interactive AI/ML project built with Python + Streamlit.

Pick a dataset and a model, tune it, see accuracy and a confusion matrix,
and make live predictions. All datasets ship with scikit-learn, so no
downloads are needed.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="ML Playground", page_icon="🤖", layout="wide")

DATASETS = {"Iris (flowers)": load_iris, "Wine (chemistry)": load_wine,
            "Breast Cancer (diagnostic)": load_breast_cancer}


@st.cache_data
def get_data(name):
    ds = DATASETS[name]()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target)
    return X, y, list(ds.target_names)


def build_model(name, p):
    if name == "k-Nearest Neighbours":
        clf = KNeighborsClassifier(n_neighbors=p["k"])
    elif name == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=p["depth"], random_state=42)
    elif name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=p["trees"], max_depth=p["depth"], random_state=42)
    else:
        clf = LogisticRegression(C=p["C"], max_iter=2000)
    return make_pipeline(StandardScaler(), clf)


# ---------- Sidebar ----------
st.sidebar.title("⚙️ Controls")
ds_name = st.sidebar.selectbox("Dataset", list(DATASETS))
model_name = st.sidebar.selectbox(
    "Model", ["k-Nearest Neighbours", "Decision Tree", "Random Forest", "Logistic Regression"])
test_size = st.sidebar.slider("Test set size (%)", 10, 50, 25) / 100

params = {}
if model_name == "k-Nearest Neighbours":
    params["k"] = st.sidebar.slider("k (neighbours)", 1, 25, 5)
elif model_name == "Decision Tree":
    params["depth"] = st.sidebar.slider("Max depth", 1, 15, 4)
elif model_name == "Random Forest":
    params["trees"] = st.sidebar.slider("Number of trees", 10, 300, 100, step=10)
    params["depth"] = st.sidebar.slider("Max depth", 1, 15, 6)
else:
    params["C"] = st.sidebar.select_slider("C (regularisation)", [0.01, 0.1, 1, 10, 100], value=1)

# ---------- Train ----------
X, y, classes = get_data(ds_name)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
model = build_model(model_name, params).fit(X_tr, y_tr)
pred = model.predict(X_te)
acc = accuracy_score(y_te, pred)
cv = cross_val_score(build_model(model_name, params), X, y, cv=5)

# ---------- Page ----------
st.title("🤖 ML Playground")
st.caption("Train a machine learning model in your browser: choose data, choose a model, tune it, and predict.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🏋️ Results", "🔮 Predict", "ℹ️ About"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", len(X))
    c2.metric("Features", X.shape[1])
    c3.metric("Classes", len(classes))
    st.dataframe(X.assign(target=[classes[i] for i in y]).head(20), use_container_width=True)
    st.subheader("Class balance")
    st.bar_chart(pd.Series([classes[i] for i in y]).value_counts())
    fx = st.selectbox("Explore a feature", X.columns)
    fig, ax = plt.subplots()
    for i, cname in enumerate(classes):
        ax.hist(X[fx][y == i], alpha=0.6, label=cname, bins=15)
    ax.set_xlabel(fx); ax.legend()
    st.pyplot(fig)

with tab2:
    c1, c2 = st.columns(2)
    c1.metric("Test accuracy", f"{acc:.1%}")
    c2.metric("5-fold CV accuracy", f"{cv.mean():.1%}", f"± {cv.std():.1%}", delta_color="off")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_te, pred), display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    plt.xticks(rotation=30)
    st.pyplot(fig)
    clf = model[-1]
    imp = getattr(clf, "feature_importances_", None)
    if imp is not None:
        st.subheader("Feature importance")
        st.bar_chart(pd.Series(imp, index=X.columns).sort_values())

with tab3:
    st.write("Set the feature values and the trained model will predict the class.")
    vals, cols = {}, st.columns(3)
    for i, f in enumerate(X.columns):
        lo, hi, mean = float(X[f].min()), float(X[f].max()), float(X[f].mean())
        vals[f] = cols[i % 3].slider(f, lo, hi, mean, step=(hi - lo) / 100)
    row = pd.DataFrame([vals])
    proba = model.predict_proba(row)[0]
    st.success(f"Prediction: **{classes[int(np.argmax(proba))]}**")
    st.bar_chart(pd.Series(proba, index=classes), y_label="Probability")

with tab4:
    st.markdown("""
**Pipeline:** data → train/test split → standard scaling → model → evaluation.

**Models:** k-NN, Decision Tree, Random Forest, Logistic Regression (scikit-learn).

**Metrics:** accuracy on a held-out test set, plus 5-fold cross-validation for a more reliable estimate.

**Stack:** Python, Streamlit, scikit-learn, pandas, NumPy, Matplotlib.

*Built by Kishore K · CSE B, 3rd year.*
""")
