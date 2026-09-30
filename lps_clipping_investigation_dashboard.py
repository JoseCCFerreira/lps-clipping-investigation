from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "lps_geometry_clipping_analysis.db"
LEAKAGE_PATH = PROJECT_DIR / "lps_leakage_results.csv"
LSL = 9.80
USL = 10.20
TARGET = (LSL + USL) / 2
IDEAL_CAPABILITY = 1.33
IDEAL_SIGMA = (USL - LSL) / (6 * IDEAL_CAPABILITY)

st.set_page_config(page_title="LPS Clipping Investigation", layout="wide")
st.title("LPS Clipping Investigation")
st.caption("Geometry, clipping performance, leakage outcomes and process capability")


@st.cache_data
def read_sql(query):
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(query, connection)


@st.cache_data
def read_leakage(path):
    return pd.read_csv(path)


data = read_sql("SELECT * FROM silver_lps_geometry")
leakage = read_leakage(LEAKAGE_PATH)
joined = data.merge(leakage, on="measurement_id", how="left", validate="one_to_one")
machine_summary = read_sql("SELECT * FROM gold_lps_machine_summary")
machine_cavity = read_sql("SELECT * FROM gold_lps_machine_cavity_summary")
stats_table = read_sql("SELECT * FROM gold_python_statistics")
findings = read_sql("SELECT * FROM gold_problem_solving_findings")

st.sidebar.header("Filters")
machines = st.sidebar.multiselect(
    "Machine", sorted(joined.machine_id.unique()), default=sorted(joined.machine_id.unique())
)
cavities = st.sidebar.multiselect(
    "Cavity", sorted(joined.cavity.unique()), default=sorted(joined.cavity.unique())
)
statuses = st.sidebar.multiselect("Clip result", ["OK", "FAIL"], default=["OK", "FAIL"])

cohort = joined[joined.machine_id.isin(machines) & joined.cavity.isin(cavities)]
filtered = cohort[cohort.clip_status.isin(statuses)]

st.header("1. Process overview")
st.write(
    "Investigate where clipping failures are concentrated and whether geometry, "
    "machine/cavity differences and leakage outcomes support a dimensional root-cause hypothesis."
)

failure_rate = 1 - filtered.clip_ok.mean() if not filtered.empty else np.nan
c1, c2, c3, c4 = st.columns(4)
c1.metric("Measurements", f"{len(filtered):,}")
c2.metric("Mean geometry (mm)", f"{filtered.geometry_mm.mean():.4f}" if not filtered.empty else "N/A")
c3.metric("Mean clipping gap (mm)", f"{filtered.clipping_gap_mm.mean():.4f}" if not filtered.empty else "N/A")
c4.metric("Clipping failure rate", f"{failure_rate:.2%}" if pd.notna(failure_rate) else "N/A")

st.header("2. Visual investigation")
if filtered.empty:
    st.info("No measurements match the selected filters.")
else:
    fig = px.box(
        filtered,
        x="machine_id",
        y="geometry_mm",
        color="clip_status",
        title="Geometry by Machine and Clip Result",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        filtered,
        x="geometry_mm",
        y="clipping_gap_mm",
        color="clip_status",
        symbol="machine_id",
        hover_data=["measurement_id", "cavity"],
        title="Geometry vs Clipping Gap",
    )
    st.plotly_chart(fig, use_container_width=True)

st.header("3. Process capability: Cp and Cpk")
st.caption(
    f"Training limits: LSL={LSL:.2f} mm, USL={USL:.2f} mm, target={TARGET:.2f} mm. "
    "Replace these limits with approved engineering specifications."
)

real_values = filtered.geometry_mm.dropna()
if len(real_values) >= 2:
    real_mean = real_values.mean()
    real_sigma = real_values.std(ddof=1)
    real_cp = (USL - LSL) / (6 * real_sigma) if real_sigma > 0 else np.nan
    real_cpk = (
        min((USL - real_mean) / (3 * real_sigma), (real_mean - LSL) / (3 * real_sigma))
        if real_sigma > 0
        else np.nan
    )
else:
    real_mean = real_values.mean() if len(real_values) else np.nan
    real_sigma = real_cp = real_cpk = np.nan

capability_comparison = pd.DataFrame(
    [
        {
            "Scenario": "Real selected measurements",
            "n": len(real_values),
            "Mean (mm)": real_mean,
            "Sigma (mm)": real_sigma,
            "Deviation from target (mm)": real_mean - TARGET,
            "Cp": real_cp,
            "Cpk": real_cpk,
        },
        {
            "Scenario": "Ideal normal benchmark",
            "n": np.nan,
            "Mean (mm)": TARGET,
            "Sigma (mm)": IDEAL_SIGMA,
            "Deviation from target (mm)": 0.0,
            "Cp": IDEAL_CAPABILITY,
            "Cpk": IDEAL_CAPABILITY,
        },
    ]
).set_index("Scenario")
st.dataframe(capability_comparison, use_container_width=True)
st.caption(
    "The ideal benchmark assumes a normal distribution centered on target with Cp=Cpk=1.33 "
    f"(sigma={IDEAL_SIGMA:.4f} mm). It is a theoretical reference, not measured or simulated data; "
    "1.33 is illustrative, not a universal requirement."
)

if len(real_values):
    figure = px.histogram(
        x=real_values,
        nbins=30,
        histnorm="probability density",
        opacity=0.45,
        labels={"x": "Geometry (mm)", "y": "Density"},
        title="Real Geometry Distribution vs Ideal Capability Benchmark",
    )
    figure.update_traces(name="Real measurements", showlegend=True)
    x_min = min(real_values.min(), LSL, TARGET - 4 * IDEAL_SIGMA)
    x_max = max(real_values.max(), USL, TARGET + 4 * IDEAL_SIGMA)
    x_grid = np.linspace(x_min, x_max, 400)
    if len(real_values) > 1 and real_values.nunique() > 1:
        real_kde = stats.gaussian_kde(real_values.to_numpy())(x_grid)
        figure.add_trace(
            go.Scatter(x=x_grid, y=real_kde, mode="lines", name="Real KDE", line={"color": "#176B87"})
        )
    ideal_density = stats.norm.pdf(x_grid, loc=TARGET, scale=IDEAL_SIGMA)
    figure.add_trace(
        go.Scatter(
            x=x_grid,
            y=ideal_density,
            mode="lines",
            name="Ideal normal (Cp=Cpk=1.33)",
            line={"color": "#D87532", "width": 3},
        )
    )
    for limit, label, color in [
        (LSL, "LSL", "#444444"),
        (TARGET, "Target", "#27864B"),
        (USL, "USL", "#444444"),
    ]:
        figure.add_vline(x=limit, line_dash="dash", line_color=color, annotation_text=label)
    st.plotly_chart(figure, use_container_width=True)
else:
    st.info("Select at least one measurement to display the distribution.")

st.header("4. Clipping and leakage association")
st.caption("Uses both clipping outcomes for the selected machines and cavities, independent of the clip-result display filter.")
if cohort.empty:
    st.info("Select at least one machine and cavity to calculate the association.")
elif cohort.clip_status.nunique() < 2 or cohort.leakage_status.nunique() < 2:
    st.info("The selected machines and cavities need both outcomes for a 2×2 association test.")
else:
    contingency = pd.crosstab(cohort.clip_status, cohort.leakage_status).reindex(
        index=["FAIL", "OK"], columns=["FAIL", "OK"], fill_value=0
    )
    chi2, chi_p, _, _ = stats.chi2_contingency(contingency)
    phi = np.sqrt(chi2 / contingency.to_numpy().sum())
    odds_ratio, fisher_p = stats.fisher_exact(contingency.to_numpy())
    st.dataframe(contingency, use_container_width=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Chi-square p-value", f"{chi_p:.3g}")
    m2.metric("Phi", f"{phi:.3f}")
    m3.metric("Odds ratio", f"{odds_ratio:.2f}")
    m4.metric("Fisher exact p-value", f"{fisher_p:.3g}")
    st.markdown(
        "A small p-value is evidence against independence, not the probability that the null hypothesis is true. "
        "The odds ratio compares odds, not failure probabilities. Association does not establish causation; "
        "shared machine, cavity, process or measurement factors may affect both outcomes."
    )

st.header("5. SQL analytical tables")
st.subheader("Machine summary")
st.dataframe(machine_summary, use_container_width=True)
st.subheader("Machine × cavity drill-down")
st.dataframe(machine_cavity.sort_values("clip_failure_rate", ascending=False), use_container_width=True)

st.header("6. Python statistical evidence")
st.dataframe(stats_table, use_container_width=True)

st.header("7. Problem-solving evidence")
st.dataframe(findings, use_container_width=True)

st.header("8. Conclusion and next steps")
if pd.notna(real_cp) and pd.notna(real_cpk):
    st.markdown(
        f"For the current selection, **Cp={real_cp:.3f}** and **Cpk={real_cpk:.3f}**, "
        f"with a mean deviation from target of **{real_mean - TARGET:+.4f} mm**. "
        + ("Cp below 1 indicates the estimated spread is wider than the tolerance band. " if real_cp < 1 else "")
        + ("Cpk below Cp indicates that off-centering further reduces capability. " if real_cpk < real_cp else "")
        + "The illustrative ideal benchmark is Cp=Cpk=1.33. Verify stability and measurement-system adequacy before treating capability as a production claim."
    )
st.markdown(
    "Next, inspect the highest-risk machine/cavity combinations, add process-parameter and chronological data, "
    "validate the measurement system, replace training limits with approved specifications, and confirm the mechanism with controlled trials or DOE."
)
