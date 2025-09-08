#!/usr/bin/env python3
"""
Streamlit dashboard to explore mining reports and sample data patterns.

Run:
  streamlit run scripts/dashboard.py
"""

import pandas as pd
import streamlit as st
from pathlib import Path


def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def main() -> None:
    st.set_page_config(page_title="Bank Dataset Insights", layout="wide")
    st.title("Bank Dataset: Patterns & Insights")

    reports_dir = Path("reports")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Univariate AUC")
        df = load_csv(reports_dir / "univariate_auc.csv")
        if df is not None and not df.empty:
            top_n = st.slider("Top-N (AUC)", 10, 100, 30, 5)
            st.dataframe(df.head(top_n))
        else:
            st.info("univariate_auc.csv not found")

        st.subheader("Mutual Information")
        df = load_csv(reports_dir / "mutual_info.csv")
        if df is not None and not df.empty:
            top_n = st.slider("Top-N (MI)", 10, 100, 30, 5)
            st.dataframe(df.head(top_n))
        else:
            st.info("mutual_info.csv not found")

    with col2:
        st.subheader("Target Correlation (|corr|)")
        df = load_csv(reports_dir / "target_correlations.csv")
        if df is not None and not df.empty:
            top_n = st.slider("Top-N (|corr|)", 10, 100, 30, 5)
            st.dataframe(df.head(top_n))
        else:
            st.info("target_correlations.csv not found")

        st.subheader("Model Importance & SHAP")
        imp = load_csv(reports_dir / "model_importance.csv")
        shap = load_csv(reports_dir / "shap_importance.csv")
        tabs = st.tabs(["Model Importance", "SHAP Importance"])
        with tabs[0]:
            if imp is not None and not imp.empty:
                top_n = st.slider("Top-N (Importance)", 10, 100, 30, 5)
                st.dataframe(imp.head(top_n))
            else:
                st.info("model_importance.csv not found")
        with tabs[1]:
            if shap is not None and not shap.empty:
                top_n = st.slider("Top-N (SHAP)", 10, 100, 30, 5)
                st.dataframe(shap.head(top_n))
            else:
                st.info("shap_importance.csv not found")

    st.caption("Use scripts/visualize_reports.py to render PNG charts into reports/figures.")


if __name__ == "__main__":
    main()


