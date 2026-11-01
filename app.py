"""
Intelligent Network Traffic Mining & Cyber Threat Prediction System
Algorithms: K-Means, Decision Tree, Linear Regression, Isolation Forest
Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, r2_score)


# ---------------------------------------------------------------
# 1. SAMPLE DATASET (synthetic network traffic)
# ---------------------------------------------------------------
def make_sample_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)

    def block(size, label, dur, src, dst, pkts, fails, cpm, ports):
        return pd.DataFrame({
            "duration":      rng.exponential(dur, size),
            "src_bytes":     rng.normal(src, src * 0.3, size).clip(50),
            "dst_bytes":     rng.normal(dst, dst * 0.3, size).clip(50),
            "packets":       rng.poisson(pkts, size),
            "failed_logins": rng.poisson(fails, size),
            "conn_per_min":  rng.normal(cpm, cpm * 0.2, size).clip(1),
            "unique_ports":  rng.poisson(ports, size),
            "label":         label,
        })

    n_normal = int(n * 0.80)
    n_each = (n - n_normal) // 3
    df = pd.concat([
        block(n_normal, "normal",     5, 2000, 3000, 40,  0.1, 20,  3),
        block(n_each,   "dos",        1, 20000, 500, 400, 0.1, 150, 1),
        block(n_each,   "portscan",   1, 100,   100, 60,  0.1, 80,  60),
        block(n_each,   "bruteforce", 2, 600,   400, 50,  15,  40,  2),
    ], ignore_index=True)

    # numeric "risk score" used by Linear Regression
    df["threat_score"] = (2 * df["failed_logins"] + 1.5 * df["unique_ports"]
                          + 0.4 * df["conn_per_min"]
                          + rng.normal(0, 3, len(df))).clip(0, 100).round(2)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def pca_plot(X_scaled, colors, title, cmap="viridis"):
    pts = PCA(n_components=2).fit_transform(X_scaled)
    fig, ax = plt.subplots()
    ax.scatter(pts[:, 0], pts[:, 1], c=colors, cmap=cmap, s=15, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    return fig


# ---------------------------------------------------------------
# 2. PAGE SETUP + DATA LOADING
# ---------------------------------------------------------------
st.set_page_config(page_title="Network Threat Miner", layout="wide")
st.title("🛡️ Intelligent Network Traffic Mining & Cyber Threat Prediction")

st.sidebar.header("1. Choose data")
source = st.sidebar.radio("Data source", ["Use sample dataset", "Upload my CSV"])

if source == "Upload my CSV":
    uploaded = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    if uploaded is None:
        st.info("Upload a CSV from the sidebar, or switch to the sample dataset.")
        st.stop()
    df = pd.read_csv(uploaded)
else:
    df = make_sample_data()
    st.sidebar.download_button("⬇️ Download sample CSV",
                               df.to_csv(index=False),
                               "sample_traffic.csv", "text/csv")

# Basic cleaning: fill missing numeric values with the median
num_cols = df.select_dtypes(include="number").columns.tolist()
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df = df.dropna().reset_index(drop=True)

if len(num_cols) < 2:
    st.error("Your data needs at least 2 numeric columns.")
    st.stop()

tabs = st.tabs(["📄 Data", "🔵 K-Means", "🌳 Decision Tree",
                "📈 Linear Regression", "🚨 Isolation Forest"])


# ---------------------------------------------------------------
# TAB 1: DATA OVERVIEW
# ---------------------------------------------------------------
with tabs[0]:
    st.subheader("Dataset preview")
    st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")
    st.dataframe(df.head(50))
    st.subheader("Summary statistics")
    st.dataframe(df.describe())
    if "label" in df.columns:
        st.subheader("Traffic type counts")
        st.bar_chart(df["label"].value_counts())


# ---------------------------------------------------------------
# TAB 2: K-MEANS (group similar traffic)
# ---------------------------------------------------------------
with tabs[1]:
    st.subheader("K-Means: group similar traffic patterns")
    feats = st.multiselect("Features", num_cols, default=num_cols, key="km_f")
    k = st.slider("Number of clusters (k)", 2, 8, 4)

    if len(feats) >= 2:
        X = StandardScaler().fit_transform(df[feats])
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        out = df.copy()
        out["cluster"] = km.labels_

        c1, c2 = st.columns(2)
        c1.pyplot(pca_plot(X, km.labels_, "Clusters (2D view)"))
        c2.write("**Cluster sizes**")
        c2.dataframe(out["cluster"].value_counts().sort_index())
        c2.write("**Average value of each feature per cluster**")
        c2.dataframe(out.groupby("cluster")[feats].mean().round(2))

        if "label" in out.columns:
            st.write("**Clusters vs. real labels**")
            st.dataframe(pd.crosstab(out["cluster"], out["label"]))
    else:
        st.warning("Pick at least 2 features.")


# ---------------------------------------------------------------
# TAB 3: DECISION TREE (classify + predict attack type)
# ---------------------------------------------------------------
with tabs[2]:
    st.subheader("Decision Tree: classify traffic as normal or attack")
    all_cols = df.columns.tolist()
    default_t = all_cols.index("label") if "label" in all_cols else len(all_cols) - 1
    target = st.selectbox("Target column (what to predict)", all_cols,
                          index=default_t, key="dt_t")
    cands = [c for c in num_cols if c != target]
    feats = st.multiselect("Features", cands, default=cands, key="dt_f")
    depth = st.slider("Max tree depth", 1, 10, 4)

    if feats and df[target].nunique() <= 20:
        y = df[target].astype(str)
        X_tr, X_te, y_tr, y_te = train_test_split(
            df[feats], y, test_size=0.25, random_state=42)
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_tr, y_tr)
        pred = tree.predict(X_te)

        st.metric("Test accuracy", f"{accuracy_score(y_te, pred):.2%}")
        st.text(classification_report(y_te, pred, zero_division=0))

        imp = pd.Series(tree.feature_importances_, index=feats).sort_values()
        st.write("**Which features matter most?**")
        st.bar_chart(imp)

        fig, ax = plt.subplots(figsize=(14, 6))
        plot_tree(tree, feature_names=feats, class_names=list(tree.classes_),
                  filled=True, ax=ax, fontsize=8)
        st.pyplot(fig)

        st.write("### 🔮 Predict a new connection")
        cols = st.columns(3)
        vals = {f: cols[i % 3].number_input(f, value=float(df[f].median()))
                for i, f in enumerate(feats)}
        if st.button("Predict threat type"):
            result = tree.predict(pd.DataFrame([vals]))[0]
            if result == "normal":
                st.success(f"Prediction: {result}")
            else:
                st.error(f"⚠️ Prediction: {result}")
    elif feats:
        st.warning("Target has too many unique values. Choose a category column like 'label'.")


# ---------------------------------------------------------------
# TAB 4: LINEAR REGRESSION (predict a numeric risk value)
# ---------------------------------------------------------------
with tabs[3]:
    st.subheader("Linear Regression: predict a numeric threat/risk value")
    default_r = num_cols.index("threat_score") if "threat_score" in num_cols else len(num_cols) - 1
    target = st.selectbox("Numeric target", num_cols, index=default_r, key="lr_t")
    cands = [c for c in num_cols if c != target]
    feats = st.multiselect("Features", cands, default=cands, key="lr_f")

    if feats:
        X_tr, X_te, y_tr, y_te = train_test_split(
            df[feats], df[target], test_size=0.25, random_state=42)
        lr = LinearRegression().fit(X_tr, y_tr)
        pred = lr.predict(X_te)

        m1, m2 = st.columns(2)
        m1.metric("R² score (1.0 = perfect)", f"{r2_score(y_te, pred):.3f}")
        m2.metric("RMSE (average error)", f"{np.sqrt(mean_squared_error(y_te, pred)):.3f}")

        fig, ax = plt.subplots()
        ax.scatter(y_te, pred, s=12, alpha=0.6)
        lo, hi = float(y_te.min()), float(y_te.max())
        ax.plot([lo, hi], [lo, hi], "r--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        st.write("**Coefficients** (how much each feature pushes the target up or down)")
        st.dataframe(pd.DataFrame({"feature": feats, "coefficient": lr.coef_}).round(4))


# ---------------------------------------------------------------
# TAB 5: ISOLATION FOREST (find unusual traffic)
# ---------------------------------------------------------------
with tabs[4]:
    st.subheader("Isolation Forest: detect anomalous traffic")
    feats = st.multiselect("Features", num_cols, default=num_cols, key="if_f")
    contam = st.slider("Expected share of anomalies", 0.01, 0.30, 0.10, 0.01)

    if len(feats) >= 2:
        X = StandardScaler().fit_transform(df[feats])
        iso = IsolationForest(contamination=contam, random_state=42).fit(X)
        out = df.copy()
        out["is_anomaly"] = (iso.predict(X) == -1).astype(int)
        out["anomaly_score"] = -iso.score_samples(X)  # higher = more unusual

        st.metric("Anomalies found", int(out["is_anomaly"].sum()))
        c1, c2 = st.columns(2)
        c1.pyplot(pca_plot(X, out["is_anomaly"], "Red = anomaly", cmap="coolwarm"))
        top = out.sort_values("anomaly_score", ascending=False).head(15)
        c2.write("**Top 15 most suspicious connections**")
        c2.dataframe(top)

        if "label" in out.columns:
            st.write("**Anomalies vs. real labels**")
            st.dataframe(pd.crosstab(out["is_anomaly"], out["label"]))

        st.download_button("⬇️ Download results as CSV",
                           out.to_csv(index=False), "anomaly_results.csv", "text/csv")
    else:
        st.warning("Pick at least 2 features.")