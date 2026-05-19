import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import wilcoxon, mannwhitneyu


# ============================================================
# Input / Output
# ============================================================

INPUT_DIR = "ATD-DATA/SELF-FIXED/METRICS/FINAL/PER-BARIS"

OUTPUT_DIR = "ATD-DATA/SELF-FIXED/METRICS/FINAL/RQ1_RQ2_ISSUE_LEVEL_WILCOXON_AND_BOXPLOT"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATTERN = "*_PER_BARIS.csv"

CHUNKSIZE = 10_000


# ============================================================
# Filtering configuration
# ============================================================

# Use "viomod" if the analysis is restricted to violation of modularity.
# Use None if you want to include all indicators.
FILTER_INDICATOR = "viomod"


# ============================================================
# Publication-ready colors
# ============================================================

PHASE_COLORS = {
    "Introduction": "#4C72B0",  # muted blue
    "Payment": "#DD8452",       # muted orange
}

EDGE_COLOR = "#333333"
GRID_COLOR = "#D9D9D9"
SEPARATOR_COLOR = "#666666"


# ============================================================
# Metrics
# ============================================================

METRICS = {
    "Fan-In": {
        "rq": "RQ1",
        "intro_file_col": "intro_file_fan_in",
        "payment_file_col": "payment_file_fan_in",
        "intro_issue_col": "intro_issue_fan_in_sum",
        "payment_issue_col": "payment_issue_fan_in_sum",
        "delta_issue_col": "delta_issue_fan_in_sum",
        "short": "Fan-In",
    },
    "Fan-Out": {
        "rq": "RQ1",
        "intro_file_col": "intro_file_fan_out",
        "payment_file_col": "payment_file_fan_out",
        "intro_issue_col": "intro_issue_fan_out_sum",
        "payment_issue_col": "payment_issue_fan_out_sum",
        "delta_issue_col": "delta_issue_fan_out_sum",
        "short": "Fan-Out",
    },
    "Cyclomatic Complexity": {
        "rq": "RQ2",
        "intro_file_col": "intro_file_cyclomatic_complexity_sum",
        "payment_file_col": "payment_file_cyclomatic_complexity_sum",
        "intro_issue_col": "intro_issue_cyclomatic_complexity_sum",
        "payment_issue_col": "payment_issue_cyclomatic_complexity_sum",
        "delta_issue_col": "delta_issue_cyclomatic_complexity_sum",
        "short": "Cyclomatic",
    },
    "Cognitive Complexity": {
        "rq": "RQ2",
        "intro_file_col": "intro_file_cognitive_complexity_sum",
        "payment_file_col": "payment_file_cognitive_complexity_sum",
        "intro_issue_col": "intro_issue_cognitive_complexity_sum",
        "payment_issue_col": "payment_issue_cognitive_complexity_sum",
        "delta_issue_col": "delta_issue_cognitive_complexity_sum",
        "short": "Cognitive",
    },
}


FIGURE_GROUPS = {
    "self_fixed_dependency_metrics": {
        "group": "Self-Fixed",
        "metrics": ["Fan-In", "Fan-Out"],
        "title": "Self-Fixed ATD: Dependency Metrics",
        "ylabel": "Issue-level dependency value",
    },
    "self_fixed_complexity_metrics": {
        "group": "Self-Fixed",
        "metrics": ["Cyclomatic Complexity", "Cognitive Complexity"],
        "title": "Self-Fixed ATD: Complexity Metrics",
        "ylabel": "Issue-level complexity value",
    },
    "non_self_fixed_dependency_metrics": {
        "group": "Non-Self-Fixed",
        "metrics": ["Fan-In", "Fan-Out"],
        "title": "Non-Self-Fixed ATD: Dependency Metrics",
        "ylabel": "Issue-level dependency value",
    },
    "non_self_fixed_complexity_metrics": {
        "group": "Non-Self-Fixed",
        "metrics": ["Cyclomatic Complexity", "Cognitive Complexity"],
        "title": "Non-Self-Fixed ATD: Complexity Metrics",
        "ylabel": "Issue-level complexity value",
    },
}


# ============================================================
# Helper functions
# ============================================================

def normalize_self_fixed_series(series):
    s = series.astype(str).str.strip().str.lower()

    result = pd.Series(np.nan, index=series.index, dtype="object")

    result[s.isin(["true", "1", "yes", "y", "self-fixed", "self_fixed"])] = "Self-Fixed"
    result[s.isin(["false", "0", "no", "n", "non-self-fixed", "non_self_fixed"])] = "Non-Self-Fixed"

    return result


def median_iqr(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()

    if len(values) == 0:
        return "NA"

    median = values.median()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    return f"{median:.2f} [{iqr:.2f}]"


def direction_from_delta(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()

    if len(values) == 0:
        return "NA"

    median_delta = values.median()

    if median_delta > 0:
        return "Increase"
    elif median_delta < 0:
        return "Decrease"
    else:
        return "No Change"


def significance_label(p, alpha=0.05):
    if pd.isna(p):
        return "NA"
    return "Significant" if p < alpha else "Not Significant"


def rank_biserial_from_wilcoxon(intro_values, payment_values):
    """
    Matched-pairs rank-biserial correlation.

    Positive value means Payment tends to be higher than Introduction.
    Negative value means Payment tends to be lower than Introduction.
    """
    intro_values = np.asarray(intro_values, dtype=float)
    payment_values = np.asarray(payment_values, dtype=float)

    diff = payment_values - intro_values
    nonzero_diff = diff[diff != 0]

    if len(nonzero_diff) == 0:
        return 0.0, "No Change"

    abs_diff = np.abs(nonzero_diff)
    ranks = pd.Series(abs_diff).rank(method="average").to_numpy()

    positive_rank_sum = ranks[nonzero_diff > 0].sum()
    negative_rank_sum = ranks[nonzero_diff < 0].sum()

    total_rank_sum = positive_rank_sum + negative_rank_sum

    if total_rank_sum == 0:
        return np.nan, "NA"

    rbc = (positive_rank_sum - negative_rank_sum) / total_rank_sum

    abs_rbc = abs(rbc)

    if abs_rbc < 0.147:
        effect_size = "Negligible"
    elif abs_rbc < 0.330:
        effect_size = "Small"
    elif abs_rbc < 0.474:
        effect_size = "Medium"
    else:
        effect_size = "Large"

    return rbc, effect_size


def wilcoxon_result(intro_values, payment_values):
    """
    Wilcoxon signed-rank test for paired issue-level observations.
    """
    paired_df = pd.DataFrame({
        "intro": pd.to_numeric(pd.Series(intro_values), errors="coerce"),
        "payment": pd.to_numeric(pd.Series(payment_values), errors="coerce"),
    }).dropna()

    if len(paired_df) == 0:
        return np.nan, np.nan, np.nan, "NA", 0

    diff = paired_df["payment"] - paired_df["intro"]

    if (diff == 0).all():
        return 0.0, 1.0, 0.0, "No Change", len(paired_df)

    stat, p_value = wilcoxon(
        paired_df["payment"],
        paired_df["intro"],
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        mode="auto"
    )

    rbc, effect_size = rank_biserial_from_wilcoxon(
        paired_df["intro"],
        paired_df["payment"]
    )

    return stat, p_value, rbc, effect_size, len(paired_df)


def mann_whitney_result(x, y):
    """
    Optional: compare deltas between Self-Fixed and Non-Self-Fixed.
    Positive rank-biserial means x tends to be larger than y.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy()

    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan, np.nan, "NA"

    u_stat, p_value = mannwhitneyu(
        x,
        y,
        alternative="two-sided",
        method="asymptotic"
    )

    rank_biserial = (2 * u_stat) / (len(x) * len(y)) - 1

    abs_rb = abs(rank_biserial)

    if abs_rb < 0.147:
        effect_size = "Negligible"
    elif abs_rb < 0.330:
        effect_size = "Small"
    elif abs_rb < 0.474:
        effect_size = "Medium"
    else:
        effect_size = "Large"

    return u_stat, p_value, rank_biserial, effect_size


def adjust_p_values(df, p_col="p_value"):
    """
    Add Bonferroni and Holm corrected p-values.
    """
    df = df.copy()

    valid_mask = ~df[p_col].isna()
    p_values = df.loc[valid_mask, p_col].astype(float).values
    m = len(p_values)

    df["p_bonferroni"] = np.nan
    df["p_holm"] = np.nan

    if m == 0:
        return df

    # Bonferroni
    df.loc[valid_mask, "p_bonferroni"] = np.minimum(p_values * m, 1.0)

    # Holm
    valid_indices = df.loc[valid_mask].index.to_list()
    sorted_order = np.argsort(p_values)

    sorted_p = p_values[sorted_order]
    sorted_indices = [valid_indices[i] for i in sorted_order]

    holm_values = []

    for i, p in enumerate(sorted_p):
        holm_values.append(min((m - i) * p, 1.0))

    # Enforce monotonicity
    for i in range(1, len(holm_values)):
        holm_values[i] = max(holm_values[i], holm_values[i - 1])

    for idx, holm_p in zip(sorted_indices, holm_values):
        df.loc[idx, "p_holm"] = holm_p

    return df


# ============================================================
# Step 1: Build issue-level dataset
# ============================================================

def build_issue_level_dataset_memory_safe():
    """
    Build issue-level dataset from file-level PER-BARIS CSV files.

    Important:
    - Only Java files are used.
    - All available Java files are aggregated.
    - File-level pairing is not required.
    - Pairing happens later at issue level.
    """

    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, CSV_PATTERN)))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

    print(f"Found CSV files: {len(csv_files)}")

    required_cols = [
        "Key",
        "indicator",
        "self_fixed",
        "file_path",
    ]

    optional_cols = [
        "source_file",
    ]

    metric_file_cols = []

    for metric_info in METRICS.values():
        metric_file_cols.append(metric_info["intro_file_col"])
        metric_file_cols.append(metric_info["payment_file_col"])

    required_cols = list(dict.fromkeys(required_cols + metric_file_cols))

    chunk_aggregates = []
    project_summary_rows = []

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        print(f"\nProcessing: {filename}")

        available_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()

        usecols = [
            col for col in required_cols + optional_cols
            if col in available_cols
        ]

        missing_required = [
            col for col in required_cols
            if col not in available_cols
        ]

        if missing_required:
            raise ValueError(f"Missing required columns in {filename}: {missing_required}")

        chunks_read = 0
        total_rows_scanned = 0
        java_rows_before_indicator = 0
        java_rows_after_indicator = 0

        for chunk in pd.read_csv(
            csv_path,
            usecols=usecols,
            chunksize=CHUNKSIZE,
            low_memory=False
        ):
            chunks_read += 1
            total_rows_scanned += len(chunk)

            if "source_file" not in chunk.columns:
                chunk["source_file"] = filename

            # Keep only Java files
            chunk = chunk[
                chunk["file_path"]
                .astype(str)
                .str.lower()
                .str.endswith(".java", na=False)
            ].copy()

            if chunk.empty:
                continue

            java_rows_before_indicator += len(chunk)

            # Optional indicator filter
            if FILTER_INDICATOR is not None:
                chunk = chunk[
                    chunk["indicator"].astype(str).str.lower().str.strip()
                    == FILTER_INDICATOR.lower()
                ].copy()

            if chunk.empty:
                continue

            java_rows_after_indicator += len(chunk)

            # Normalize self-fixed group
            chunk["self_fixed_group"] = normalize_self_fixed_series(
                chunk["self_fixed"]
            )

            chunk = chunk[
                chunk["self_fixed_group"].isin(
                    ["Self-Fixed", "Non-Self-Fixed"]
                )
            ].copy()

            if chunk.empty:
                continue

            # Convert metric columns to numeric
            for col in metric_file_cols:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            # Aggregate file-level metrics into issue-level sums
            group_cols = [
                "source_file",
                "Key",
                "indicator",
                "self_fixed_group",
            ]

            agg_spec = {}

            for col in metric_file_cols:
                agg_spec[f"{col}__sum"] = (col, "sum")
                agg_spec[f"{col}__count"] = (col, "count")

            chunk_issue_agg = (
                chunk
                .groupby(group_cols, dropna=False)
                .agg(**agg_spec)
                .reset_index()
            )

            chunk_aggregates.append(chunk_issue_agg)

        project_summary_rows.append({
            "source_file": filename,
            "chunks_read": chunks_read,
            "total_rows_scanned": total_rows_scanned,
            "java_rows_before_indicator_filter": java_rows_before_indicator,
            "java_rows_after_indicator_filter": java_rows_after_indicator,
        })

        print(f"  Chunks read                       : {chunks_read}")
        print(f"  Rows scanned                      : {total_rows_scanned}")
        print(f"  Java rows before indicator filter : {java_rows_before_indicator}")
        print(f"  Java rows after indicator filter  : {java_rows_after_indicator}")

    if len(chunk_aggregates) == 0:
        raise ValueError("No issue-level aggregates were generated.")

    print("\nCombining chunk-level issue aggregates...")

    all_chunk_agg = pd.concat(chunk_aggregates, ignore_index=True)

    group_cols = [
        "source_file",
        "Key",
        "indicator",
        "self_fixed_group",
    ]

    sum_count_cols = [
        col for col in all_chunk_agg.columns
        if col.endswith("__sum") or col.endswith("__count")
    ]

    final_issue_agg = (
        all_chunk_agg
        .groupby(group_cols, dropna=False)[sum_count_cols]
        .sum()
        .reset_index()
    )

    # Create clean issue-level metric columns
    for metric_name, metric_info in METRICS.items():
        intro_file_col = metric_info["intro_file_col"]
        payment_file_col = metric_info["payment_file_col"]

        intro_issue_col = metric_info["intro_issue_col"]
        payment_issue_col = metric_info["payment_issue_col"]
        delta_issue_col = metric_info["delta_issue_col"]

        intro_sum_col = f"{intro_file_col}__sum"
        intro_count_col = f"{intro_file_col}__count"

        payment_sum_col = f"{payment_file_col}__sum"
        payment_count_col = f"{payment_file_col}__count"

        final_issue_agg[intro_issue_col] = final_issue_agg[intro_sum_col]
        final_issue_agg[payment_issue_col] = final_issue_agg[payment_sum_col]

        final_issue_agg.loc[
            final_issue_agg[intro_count_col] == 0,
            intro_issue_col
        ] = np.nan

        final_issue_agg.loc[
            final_issue_agg[payment_count_col] == 0,
            payment_issue_col
        ] = np.nan

        final_issue_agg[delta_issue_col] = (
            final_issue_agg[payment_issue_col]
            - final_issue_agg[intro_issue_col]
        )

        final_issue_agg[f"n_intro_java_files_for_{metric_name}"] = final_issue_agg[intro_count_col]
        final_issue_agg[f"n_payment_java_files_for_{metric_name}"] = final_issue_agg[payment_count_col]

    clean_cols = [
        "source_file",
        "Key",
        "indicator",
        "self_fixed_group",
    ]

    for metric_name, metric_info in METRICS.items():
        clean_cols.extend([
            metric_info["intro_issue_col"],
            metric_info["payment_issue_col"],
            metric_info["delta_issue_col"],
            f"n_intro_java_files_for_{metric_name}",
            f"n_payment_java_files_for_{metric_name}",
        ])

    issue_level_df = final_issue_agg[clean_cols].copy()

    issue_level_output = os.path.join(
        OUTPUT_DIR,
        "RQ1_RQ2_issue_level_aggregated_metrics_all_java_files.csv"
    )

    issue_level_df.to_csv(issue_level_output, index=False)

    project_summary_df = pd.DataFrame(project_summary_rows)

    project_summary_output = os.path.join(
        OUTPUT_DIR,
        "RQ1_RQ2_issue_level_java_rows_used_summary.csv"
    )

    project_summary_df.to_csv(project_summary_output, index=False)

    print("\nIssue-level aggregated dataset saved to:")
    print(issue_level_output)

    print("\nJava row summary saved to:")
    print(project_summary_output)

    return issue_level_df


# ============================================================
# Step 2: Wilcoxon statistical tests
# ============================================================

def build_descriptive_summary(issue_df):
    """
    Descriptive summary based on paired issue-level observations.
    """
    rows = []

    for metric_name, metric_info in METRICS.items():
        rq = metric_info["rq"]
        intro_col = metric_info["intro_issue_col"]
        payment_col = metric_info["payment_issue_col"]
        delta_col = metric_info["delta_issue_col"]

        for group_name in ["Self-Fixed", "Non-Self-Fixed"]:
            g = issue_df[
                issue_df["self_fixed_group"] == group_name
            ].copy()

            paired_g = g.dropna(subset=[intro_col, payment_col]).copy()

            rows.append({
                "RQ": rq,
                "Metric": metric_name,
                "Group": group_name,
                "N_Paired_Issues": len(paired_g),
                "Introduction Median [IQR]": median_iqr(paired_g[intro_col]),
                "Payment Median [IQR]": median_iqr(paired_g[payment_col]),
                "Delta Median [IQR]": median_iqr(paired_g[delta_col]),
                "Direction": direction_from_delta(paired_g[delta_col]),
                "N_Intro_Available_Issues": g[intro_col].notna().sum(),
                "N_Payment_Available_Issues": g[payment_col].notna().sum(),
            })

    return pd.DataFrame(rows)


def build_wilcoxon_intro_vs_payment_tests(issue_df):
    """
    Wilcoxon tests using the same paired issue-level observations
    later used for boxplots.
    """
    rows = []

    for metric_name, metric_info in METRICS.items():
        rq = metric_info["rq"]
        intro_col = metric_info["intro_issue_col"]
        payment_col = metric_info["payment_issue_col"]

        for group_name in ["Self-Fixed", "Non-Self-Fixed"]:
            g = issue_df[
                issue_df["self_fixed_group"] == group_name
            ].copy()

            paired_g = g.dropna(subset=[intro_col, payment_col]).copy()

            stat, p_value, rbc, effect_size, n_paired = wilcoxon_result(
                paired_g[intro_col],
                paired_g[payment_col]
            )

            rows.append({
                "RQ": rq,
                "Metric": metric_name,
                "Comparison": "Payment vs Introduction",
                "Group": group_name,
                "N_Paired_Issues": n_paired,
                "Introduction Median [IQR]": median_iqr(paired_g[intro_col]),
                "Payment Median [IQR]": median_iqr(paired_g[payment_col]),
                "Delta Median [IQR]": median_iqr(paired_g[payment_col] - paired_g[intro_col]),
                "Wilcoxon_Statistic": stat,
                "p_value": p_value,
                "Rank_Biserial_Correlation_Payment_vs_Introduction": rbc,
                "Effect_Size": effect_size,
            })

    result_df = pd.DataFrame(rows)
    result_df = adjust_p_values(result_df, p_col="p_value")

    result_df["Significance_raw_p_0.05"] = result_df["p_value"].apply(significance_label)
    result_df["Significance_holm_0.05"] = result_df["p_holm"].apply(significance_label)
    result_df["Significance_bonferroni_0.05"] = result_df["p_bonferroni"].apply(significance_label)

    return result_df


def build_delta_group_tests(issue_df):
    """
    Optional additional analysis:
    compare issue-level deltas between Self-Fixed and Non-Self-Fixed.
    """
    rows = []

    for metric_name, metric_info in METRICS.items():
        rq = metric_info["rq"]
        delta_col = metric_info["delta_issue_col"]

        self_delta = issue_df[
            issue_df["self_fixed_group"] == "Self-Fixed"
        ][delta_col].dropna()

        non_self_delta = issue_df[
            issue_df["self_fixed_group"] == "Non-Self-Fixed"
        ][delta_col].dropna()

        u_stat, p_value, rb, effect_size = mann_whitney_result(
            self_delta,
            non_self_delta
        )

        rows.append({
            "RQ": rq,
            "Metric": metric_name,
            "Comparison": "Delta Self-Fixed vs Delta Non-Self-Fixed",
            "N_Self_Fixed_Delta_Issues": len(self_delta),
            "N_Non_Self_Fixed_Delta_Issues": len(non_self_delta),
            "Self-Fixed Delta Median [IQR]": median_iqr(self_delta),
            "Non-Self-Fixed Delta Median [IQR]": median_iqr(non_self_delta),
            "Mann_Whitney_U": u_stat,
            "p_value": p_value,
            "Rank_Biserial_Correlation_Self_vs_NonSelf": rb,
            "Effect_Size": effect_size,
        })

    result_df = pd.DataFrame(rows)
    result_df = adjust_p_values(result_df, p_col="p_value")

    result_df["Significance_raw_p_0.05"] = result_df["p_value"].apply(significance_label)
    result_df["Significance_holm_0.05"] = result_df["p_holm"].apply(significance_label)
    result_df["Significance_bonferroni_0.05"] = result_df["p_bonferroni"].apply(significance_label)

    return result_df


# ============================================================
# Step 3: Boxplot data from the same paired observations
# ============================================================

def build_paired_long_format_for_boxplot(issue_df):
    """
    Build long-format boxplot data using exactly the same paired
    issue-level observations as the Wilcoxon test.

    For each metric and group:
    - issue is included only if both Introduction and Payment values exist.
    """
    rows = []

    for metric_name, metric_info in METRICS.items():
        intro_col = metric_info["intro_issue_col"]
        payment_col = metric_info["payment_issue_col"]

        metric_df = issue_df.dropna(
            subset=[
                intro_col,
                payment_col,
            ]
        ).copy()

        intro_df = metric_df[
            [
                "source_file",
                "Key",
                "indicator",
                "self_fixed_group",
                intro_col,
            ]
        ].copy()

        intro_df = intro_df.rename(columns={intro_col: "value"})
        intro_df["RQ"] = metric_info["rq"]
        intro_df["Metric"] = metric_name
        intro_df["Metric_Short"] = metric_info["short"]
        intro_df["Phase"] = "Introduction"
        intro_df["Metric_Phase"] = metric_info["short"] + "\nIntroduction"

        payment_df = metric_df[
            [
                "source_file",
                "Key",
                "indicator",
                "self_fixed_group",
                payment_col,
            ]
        ].copy()

        payment_df = payment_df.rename(columns={payment_col: "value"})
        payment_df["RQ"] = metric_info["rq"]
        payment_df["Metric"] = metric_name
        payment_df["Metric_Short"] = metric_info["short"]
        payment_df["Phase"] = "Payment"
        payment_df["Metric_Phase"] = metric_info["short"] + "\nPayment"

        rows.append(intro_df)
        rows.append(payment_df)

    long_df = pd.concat(rows, ignore_index=True)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    output_path = os.path.join(
        OUTPUT_DIR,
        "RQ1_RQ2_issue_level_boxplot_long_format_PAIRED_ONLY.csv"
    )

    long_df.to_csv(output_path, index=False)

    print("\nPaired-only boxplot long-format dataset saved to:")
    print(output_path)

    return long_df


def save_boxplot_summary(long_df):
    """
    Save summary statistics for plotted distributions.
    """
    rows = []

    for group_name in ["Self-Fixed", "Non-Self-Fixed"]:
        for metric_name in METRICS.keys():
            for phase_name in ["Introduction", "Payment"]:

                values = long_df[
                    (long_df["self_fixed_group"] == group_name) &
                    (long_df["Metric"] == metric_name) &
                    (long_df["Phase"] == phase_name)
                ]["value"].dropna()

                if len(values) == 0:
                    rows.append({
                        "Group": group_name,
                        "Metric": metric_name,
                        "Phase": phase_name,
                        "N": 0,
                        "Median": None,
                        "Q1": None,
                        "Q3": None,
                        "IQR": None,
                        "Min": None,
                        "Max": None,
                    })
                else:
                    q1 = values.quantile(0.25)
                    q3 = values.quantile(0.75)

                    rows.append({
                        "Group": group_name,
                        "Metric": metric_name,
                        "Phase": phase_name,
                        "N": len(values),
                        "Median": values.median(),
                        "Q1": q1,
                        "Q3": q3,
                        "IQR": q3 - q1,
                        "Min": values.min(),
                        "Max": values.max(),
                    })

    summary_df = pd.DataFrame(rows)

    output_path = os.path.join(
        OUTPUT_DIR,
        "RQ1_RQ2_issue_level_boxplot_summary_PAIRED_ONLY.csv"
    )

    summary_df.to_csv(output_path, index=False)

    print("\nBoxplot summary saved to:")
    print(output_path)

    return summary_df


def check_consistency_between_wilcoxon_and_boxplot(issue_df, long_df):
    """
    Confirm that boxplot N equals Wilcoxon paired N.
    """
    rows = []

    for metric_name, metric_info in METRICS.items():
        intro_col = metric_info["intro_issue_col"]
        payment_col = metric_info["payment_issue_col"]

        for group_name in ["Self-Fixed", "Non-Self-Fixed"]:

            paired_df = issue_df[
                issue_df["self_fixed_group"] == group_name
            ].dropna(
                subset=[
                    intro_col,
                    payment_col,
                ]
            )

            box_intro_n = len(
                long_df[
                    (long_df["self_fixed_group"] == group_name) &
                    (long_df["Metric"] == metric_name) &
                    (long_df["Phase"] == "Introduction")
                ]["value"].dropna()
            )

            box_payment_n = len(
                long_df[
                    (long_df["self_fixed_group"] == group_name) &
                    (long_df["Metric"] == metric_name) &
                    (long_df["Phase"] == "Payment")
                ]["value"].dropna()
            )

            rows.append({
                "Metric": metric_name,
                "Group": group_name,
                "N_Paired_Issues_Wilcoxon": len(paired_df),
                "N_Boxplot_Introduction": box_intro_n,
                "N_Boxplot_Payment": box_payment_n,
                "Consistent": len(paired_df) == box_intro_n == box_payment_n,
            })

    consistency_df = pd.DataFrame(rows)

    output_path = os.path.join(
        OUTPUT_DIR,
        "RQ1_RQ2_issue_level_wilcoxon_boxplot_consistency_check.csv"
    )

    consistency_df.to_csv(output_path, index=False)

    print("\nConsistency check saved to:")
    print(output_path)

    print("\nConsistency check:")
    print(consistency_df)

    return consistency_df


# ============================================================
# Step 4: Plotting helpers
# ============================================================

def apply_box_colors(box, categories):
    for patch, category in zip(box["boxes"], categories):
        if "Introduction" in category:
            patch.set_facecolor(PHASE_COLORS["Introduction"])
        elif "Payment" in category:
            patch.set_facecolor(PHASE_COLORS["Payment"])
        else:
            patch.set_facecolor("#BBBBBB")

        patch.set_edgecolor(EDGE_COLOR)
        patch.set_alpha(0.85)

    for median in box["medians"]:
        median.set_color("#000000")
        median.set_linewidth(2.0)

    for whisker in box["whiskers"]:
        whisker.set_color(EDGE_COLOR)
        whisker.set_linewidth(1.2)

    for cap in box["caps"]:
        cap.set_color(EDGE_COLOR)
        cap.set_linewidth(1.2)


def get_categories(metric_names):
    categories = []

    for metric_name in metric_names:
        short_name = METRICS[metric_name]["short"]
        categories.append(short_name + "\nIntroduction")
        categories.append(short_name + "\nPayment")

    return categories


def get_boxplot_data(plot_df, categories):
    data = []

    for category in categories:
        values = plot_df[
            plot_df["Metric_Phase"] == category
        ]["value"].dropna()

        data.append(values)

    return data


def get_phase_legend_handles():
    return [
        Patch(
            facecolor=PHASE_COLORS["Introduction"],
            edgecolor=EDGE_COLOR,
            label="Introduction",
            alpha=0.85
        ),
        Patch(
            facecolor=PHASE_COLORS["Payment"],
            edgecolor=EDGE_COLOR,
            label="Payment",
            alpha=0.85
        ),
    ]


def style_axis(ax, ylabel=None):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13)

    ax.set_xlabel("")

    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        color=GRID_COLOR,
        alpha=0.8
    )

    ax.axvline(
        x=2.5,
        linestyle="--",
        linewidth=1.0,
        color=SEPARATOR_COLOR,
        alpha=0.6
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_boxplot(ax, data, categories):
    box = ax.boxplot(
        data,
        labels=categories,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        medianprops={
            "linewidth": 2,
            "color": "#000000"
        },
        boxprops={
            "linewidth": 1.2,
            "color": EDGE_COLOR
        },
        whiskerprops={
            "linewidth": 1.2,
            "color": EDGE_COLOR
        },
        capprops={
            "linewidth": 1.2,
            "color": EDGE_COLOR
        }
    )

    apply_box_colors(box, categories)

    return box


def make_boxplot_for_group(long_df, figure_key, figure_info):
    group_name = figure_info["group"]
    metric_names = figure_info["metrics"]
    title = figure_info["title"]
    ylabel = figure_info["ylabel"]

    plot_df = long_df[
        (long_df["self_fixed_group"] == group_name) &
        (long_df["Metric"].isin(metric_names))
    ].copy()

    if plot_df.empty:
        print(f"Skipping {figure_key}: no data.")
        return

    categories = get_categories(metric_names)
    data = get_boxplot_data(plot_df, categories)

    fig, ax = plt.subplots(figsize=(10, 6))

    create_boxplot(ax, data, categories)

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=12
    )

    style_axis(ax, ylabel=ylabel)

    ax.legend(
        handles=get_phase_legend_handles(),
        loc="upper right",
        frameon=False,
        fontsize=11
    )

    plt.tight_layout()

    output_png = os.path.join(
        OUTPUT_DIR,
        f"{figure_key}_issue_level_boxplot_publication.png"
    )

    output_pdf = os.path.join(
        OUTPUT_DIR,
        f"{figure_key}_issue_level_boxplot_publication.pdf"
    )

    plt.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")

    plt.close()

    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")


def make_combined_four_panel_figure(long_df):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(16, 10)
    )

    axes = axes.flatten()
    figure_items = list(FIGURE_GROUPS.items())

    for ax, (_, figure_info) in zip(axes, figure_items):
        group_name = figure_info["group"]
        metric_names = figure_info["metrics"]
        title = figure_info["title"]
        ylabel = figure_info["ylabel"]

        plot_df = long_df[
            (long_df["self_fixed_group"] == group_name) &
            (long_df["Metric"].isin(metric_names))
        ].copy()

        categories = get_categories(metric_names)
        data = get_boxplot_data(plot_df, categories)

        create_boxplot(ax, data, categories)

        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold",
            pad=10
        )

        style_axis(ax, ylabel=ylabel)

        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

    fig.legend(
        handles=get_phase_legend_handles(),
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 1.02)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_png = os.path.join(
        OUTPUT_DIR,
        "RQ1_RQ2_issue_level_boxplot_grouped_by_self_fixed_combined.png"
    )

    output_pdf = os.path.join(
        OUTPUT_DIR,
        "RQ1_RQ2_issue_level_boxplot_grouped_by_self_fixed_combined.pdf"
    )

    plt.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")

    plt.close()

    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")


def make_paper_ready_two_figures(long_df):
    figure_specs = {
        "RQ1_dependency_metrics_by_self_fixed_status_publication": {
            "metrics": ["Fan-In", "Fan-Out"],
            "title": "Issue-Level Dependency Metrics",
            "ylabel": "Issue-level dependency value",
        },
        "RQ2_complexity_metrics_by_self_fixed_status_publication": {
            "metrics": ["Cyclomatic Complexity", "Cognitive Complexity"],
            "title": "Issue-Level Complexity Metrics",
            "ylabel": "Issue-level complexity value",
        },
    }

    for figure_key, figure_info in figure_specs.items():

        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(15, 5.8),
            sharey=False
        )

        for ax, group_name in zip(axes, ["Self-Fixed", "Non-Self-Fixed"]):
            metric_names = figure_info["metrics"]

            plot_df = long_df[
                (long_df["self_fixed_group"] == group_name) &
                (long_df["Metric"].isin(metric_names))
            ].copy()

            categories = get_categories(metric_names)
            data = get_boxplot_data(plot_df, categories)

            create_boxplot(ax, data, categories)

            ax.set_title(
                group_name,
                fontsize=14,
                fontweight="bold",
                pad=10
            )

            style_axis(ax, ylabel=figure_info["ylabel"])

            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)

        fig.legend(
            handles=get_phase_legend_handles(),
            loc="upper center",
            ncol=2,
            frameon=False,
            fontsize=12,
            bbox_to_anchor=(0.5, 1.02)
        )

        fig.suptitle(
            figure_info["title"],
            fontsize=15,
            fontweight="bold",
            y=1.08
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_png = os.path.join(
            OUTPUT_DIR,
            f"{figure_key}.png"
        )

        output_pdf = os.path.join(
            OUTPUT_DIR,
            f"{figure_key}.pdf"
        )

        plt.savefig(output_png, dpi=600, bbox_inches="tight")
        plt.savefig(output_pdf, bbox_inches="tight")

        plt.close()

        print(f"Saved: {output_png}")
        print(f"Saved: {output_pdf}")


# ============================================================
# Main
# ============================================================

def main():
    print("Starting combined issue-level Wilcoxon and boxplot pipeline...")

    # --------------------------------------------------------
    # 1. Build issue-level dataset
    # --------------------------------------------------------

    issue_df = build_issue_level_dataset_memory_safe()

    print("\nIssue-level dataset overview:")
    print("Total issue-level rows:", len(issue_df))
    print("Unique ATD items:", issue_df["Key"].nunique())
    print("Self-fixed group counts:")
    print(issue_df["self_fixed_group"].value_counts(dropna=False))

    if FILTER_INDICATOR is not None:
        print(f"\nIndicator filter applied: {FILTER_INDICATOR}")
    else:
        print("\nIndicator filter applied: None")

    # --------------------------------------------------------
    # 2. Statistical tests
    # --------------------------------------------------------

    print("\nBuilding descriptive summary based on paired issue-level observations...")
    descriptive_df = build_descriptive_summary(issue_df)

    descriptive_output = os.path.join(
        OUTPUT_DIR,
        "RQ1_RQ2_issue_level_descriptive_summary_wilcoxon_PAIRED_ONLY.csv"
    )

    descriptive_df.to_csv(descriptive_output, index=False)

    print("\nRunning Wilcoxon signed-rank tests...")
    wilcoxon_df = build_wilcoxon_intro_vs_payment_tests(issue_df)

    wilcoxon_output = os.path.join(
        OUTPUT_DIR,
        "RQ1_RQ2_issue_level_wilcoxon_payment_vs_introduction_PAIRED_ONLY.csv"
    )

    wilcoxon_df.to_csv(wilcoxon_output, index=False)

    print("\nRunning optional Mann-Whitney U tests for delta comparison...")
    delta_group_df = build_delta_group_tests(issue_df)

    delta_group_output = os.path.join(
        OUTPUT_DIR,
        "RQ1_RQ2_issue_level_delta_self_vs_nonself_mannwhitney.csv"
    )

    delta_group_df.to_csv(delta_group_output, index=False)

    # --------------------------------------------------------
    # 3. Boxplot data from same paired observations
    # --------------------------------------------------------

    print("\nBuilding paired-only long-format dataset for boxplots...")
    long_df = build_paired_long_format_for_boxplot(issue_df)

    print("\nSaving boxplot summary...")
    boxplot_summary_df = save_boxplot_summary(long_df)

    print("\nChecking consistency between Wilcoxon and boxplot...")
    consistency_df = check_consistency_between_wilcoxon_and_boxplot(
        issue_df=issue_df,
        long_df=long_df
    )

    # --------------------------------------------------------
    # 4. Create figures
    # --------------------------------------------------------

    print("\nCreating separate boxplots by self-fixed status...")
    for figure_key, figure_info in FIGURE_GROUPS.items():
        make_boxplot_for_group(
            long_df=long_df,
            figure_key=figure_key,
            figure_info=figure_info
        )

    print("\nCreating combined 2x2 figure...")
    make_combined_four_panel_figure(long_df)

    print("\nCreating paper-ready two-figure format...")
    make_paper_ready_two_figures(long_df)

    # --------------------------------------------------------
    # 5. Final output
    # --------------------------------------------------------

    print("\nDone.")
    print("Output folder:")
    print(OUTPUT_DIR)

    print("\nMain statistical output:")
    print(wilcoxon_output)

    print("\nMain boxplot outputs:")
    print(os.path.join(OUTPUT_DIR, "RQ1_dependency_metrics_by_self_fixed.pdf"))
    print(os.path.join(OUTPUT_DIR, "RQ2_complexity_metrics_by_self_fixed.pdf"))

    print("\nConsistency check:")
    print(consistency_df)


if __name__ == "__main__":
    main()