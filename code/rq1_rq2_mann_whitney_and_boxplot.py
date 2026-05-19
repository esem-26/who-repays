import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, rankdata
from scipy.special import log_ndtr


# ============================================================
# Input / Output
# ============================================================

INPUT_DIR = "ATD-DATA/SELF-FIXED/METRICS/FINAL/PER-BARIS"

OUTPUT_DIR = "ATD-DATA/SELF-FIXED/METRICS/FINAL/RQ1_RQ2_MANN_WHITNEY_AND_BOXPLOT"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATTERN = "*_PER_BARIS.csv"
CHUNKSIZE = 10_000

# Use "viomod" if the analysis should only include VioMod ATD items.
# Use None if the analysis should include all indicators.
FILTER_INDICATOR = "viomod"

# Optional sampling for boxplots only.
# This does NOT affect statistical tests or descriptive CSV results.
MAX_VALUES_PER_BOX = None


# ============================================================
# Publication-ready colors
# ============================================================

PHASE_COLORS = {
    "Introduction": "#4C72B0",
    "Payment": "#DD8452",
}

EDGE_COLOR = "#333333"
GRID_COLOR = "#D9D9D9"
SEPARATOR_COLOR = "#666666"


# ============================================================
# Metrics for RQ1 and RQ2
# ============================================================

METRICS = {
    "Fan-In": {
        "rq": "RQ1",
        "intro_col": "intro_file_fan_in",
        "payment_col": "payment_file_fan_in",
        "delta_col": "delta_file_fan_in",
        "ylabel": "Fan-In",
    },
    "Fan-Out": {
        "rq": "RQ1",
        "intro_col": "intro_file_fan_out",
        "payment_col": "payment_file_fan_out",
        "delta_col": "delta_file_fan_out",
        "ylabel": "Fan-Out",
    },
    "Cyclomatic Complexity": {
        "rq": "RQ2",
        "intro_col": "intro_file_cyclomatic_complexity_sum",
        "payment_col": "payment_file_cyclomatic_complexity_sum",
        "delta_col": "delta_file_cyclomatic_complexity_sum",
        "ylabel": "Cyclomatic Complexity",
    },
    "Cognitive Complexity": {
        "rq": "RQ2",
        "intro_col": "intro_file_cognitive_complexity_sum",
        "payment_col": "payment_file_cognitive_complexity_sum",
        "delta_col": "delta_file_cognitive_complexity_sum",
        "ylabel": "Cognitive Complexity",
    },
}

GROUPS = ["Self-Fixed", "Non-Self-Fixed"]


# ============================================================
# Helper functions
# ============================================================

def normalize_self_fixed_series(series):
    """
    Convert the self_fixed column into two clean groups:
    - Self-Fixed
    - Non-Self-Fixed
    """
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


def median_value(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return float(values.median())


def descriptive_numeric(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()

    if len(values) == 0:
        return {
            "N": 0,
            "Median": np.nan,
            "Q1": np.nan,
            "Q3": np.nan,
            "IQR": np.nan,
            "Min": np.nan,
            "Max": np.nan,
        }

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    return {
        "N": len(values),
        "Median": values.median(),
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
        "Min": values.min(),
        "Max": values.max(),
    }


def direction_from_delta(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()

    if len(values) == 0:
        return "NA"

    median_delta = values.median()

    if median_delta > 0:
        return "Increase"
    if median_delta < 0:
        return "Decrease"
    return "No Change"


def format_p_value(p):
    """
    Format ordinary p-value for readable reporting.
    """
    if pd.isna(p):
        return "NA"

    if p == 0.0:
        return "p < 1e-308"

    if p < 0.001:
        return f"{p:.3e}"

    return f"{p:.6f}"


def format_p_value_with_log(p, minus_log10_p=None):
    """
    Format p-value safely when p-value underflows to 0.0.

    Example:
    p_value = 0.0 and minus_log10_p = 420.7
    will be reported as:
    p < 1e-420
    """
    if pd.isna(p):
        return "NA"

    if p == 0.0:
        if minus_log10_p is not None and np.isfinite(minus_log10_p):
            exponent = int(np.floor(minus_log10_p))
            return f"p < 1e-{exponent}"
        return "p < 1e-308"

    if p < 0.001:
        return f"{p:.3e}"

    return f"{p:.6f}"


def effect_size_label_cliffs_delta(abs_delta):
    """
    Interpret Cliff's Delta using common thresholds:
    |d| < 0.147  = Negligible
    |d| < 0.330  = Small
    |d| < 0.474  = Medium
    otherwise    = Large
    """
    if pd.isna(abs_delta):
        return "NA"

    if abs_delta < 0.147:
        return "Negligible"
    elif abs_delta < 0.330:
        return "Small"
    elif abs_delta < 0.474:
        return "Medium"
    else:
        return "Large"


def effect_direction_from_signed_delta(signed_delta, x_label, y_label):
    """
    Interpret the sign of Cliff's Delta.

    x_label and y_label depend on the comparison.

    For Introduction vs Payment:
    x_label = Introduction
    y_label = Payment

    signed_delta > 0 means x tends to be larger than y.
    signed_delta < 0 means y tends to be larger than x.
    """
    if pd.isna(signed_delta):
        return "NA"

    if signed_delta > 0:
        return f"{x_label} tends to be larger than {y_label}"
    elif signed_delta < 0:
        return f"{y_label} tends to be larger than {x_label}"
    else:
        return "No dominance"


def mann_whitney_log_p_value(x, y):
    """
    Compute a numerically stable asymptotic two-sided Mann-Whitney p-value
    using log probability.

    This helps when scipy's ordinary p-value underflows to 0.0.
    The returned minus_log10_p is useful for reporting very small p-values.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy()

    n1 = len(x)
    n2 = len(y)

    if n1 == 0 or n2 == 0:
        return np.nan, np.nan

    combined = np.concatenate([x, y])
    n = n1 + n2

    ranks = rankdata(combined, method="average")
    r1 = np.sum(ranks[:n1])
    u1 = r1 - (n1 * (n1 + 1) / 2)

    mean_u = n1 * n2 / 2

    # Tie correction
    _, counts = np.unique(combined, return_counts=True)
    tie_sum = np.sum(counts**3 - counts)

    if n <= 1:
        return np.nan, np.nan

    var_u = (n1 * n2 / 12) * ((n + 1) - tie_sum / (n * (n - 1)))

    if var_u <= 0:
        return np.nan, np.nan

    z = (u1 - mean_u) / np.sqrt(var_u)

    # Two-sided log p-value:
    # log(p) = log(2 * Phi(-abs(z)))
    log_p = math.log(2) + log_ndtr(-abs(z))

    if not np.isfinite(log_p):
        return 0.0, np.inf

    minus_log10_p = -log_p / math.log(10)

    # Convert to ordinary p-value only when it is representable.
    if log_p < math.log(np.finfo(float).tiny):
        p_value = 0.0
    else:
        p_value = float(math.exp(log_p))

    return p_value, minus_log10_p


def mann_whitney_result(x, y, x_label="Group X", y_label="Group Y"):
    """
    Mann-Whitney U test for two independent samples.

    Cliff's Delta is computed from U:

        delta_signed = (2U / (n_x * n_y)) - 1

    The signed value is kept in:
        Cliffs_Delta_Signed

    The positive value is reported in:
        Cliffs_Delta = abs(Cliffs_Delta_Signed)

    This satisfies the reporting need where Cliff's Delta should not appear
    as a negative value, while preserving the actual direction separately.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy()

    if len(x) == 0 or len(y) == 0:
        return {
            "Mann_Whitney_U": np.nan,
            "p_value": np.nan,
            "minus_log10_p": np.nan,
            "p_value_report": "NA",
            "Cliffs_Delta_Signed": np.nan,
            "Cliffs_Delta": np.nan,
            "Effect_Size": "NA",
            "Effect_Direction": "NA",
        }

    u_stat, p_value = mannwhitneyu(
        x,
        y,
        alternative="two-sided",
        method="asymptotic"
    )

    _, minus_log10_p = mann_whitney_log_p_value(x, y)

    p_value_report = format_p_value_with_log(
        p=p_value,
        minus_log10_p=minus_log10_p
    )

    cliffs_delta_signed = (2 * u_stat) / (len(x) * len(y)) - 1

    # Main requested change:
    # Cliff's Delta is reported as a positive magnitude.
    cliffs_delta = abs(cliffs_delta_signed)

    effect_size = effect_size_label_cliffs_delta(cliffs_delta)

    effect_direction = effect_direction_from_signed_delta(
        signed_delta=cliffs_delta_signed,
        x_label=x_label,
        y_label=y_label,
    )

    return {
        "Mann_Whitney_U": u_stat,
        "p_value": p_value,
        "minus_log10_p": minus_log10_p,
        "p_value_report": p_value_report,
        "Cliffs_Delta_Signed": cliffs_delta_signed,
        "Cliffs_Delta": cliffs_delta,
        "Effect_Size": effect_size,
        "Effect_Direction": effect_direction,
    }


def adjust_p_values(df, p_col="p_value"):
    """
    Add Bonferroni and Holm corrected p-values.

    Notes:
    - Numeric corrected p-values may still be 0.0 when the original p-value
      is below machine precision.
    - Use the *_report columns for paper reporting.
    """
    df = df.copy()
    valid_mask = ~df[p_col].isna()
    p_values = df.loc[valid_mask, p_col].astype(float).values
    m = len(p_values)

    df["p_bonferroni"] = np.nan
    df["p_holm"] = np.nan

    if m == 0:
        return df

    # Bonferroni correction
    df.loc[valid_mask, "p_bonferroni"] = np.minimum(p_values * m, 1.0)

    # Holm correction
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


def add_p_value_report_columns(df):
    """
    Add formatted p-value columns for raw, Holm, and Bonferroni p-values.
    """
    df = df.copy()

    if "p_value" in df.columns:
        df["p_value_report"] = df.apply(
            lambda row: format_p_value_with_log(
                row["p_value"],
                row["minus_log10_p"] if "minus_log10_p" in df.columns else None
            ),
            axis=1
        )

    if "p_holm" in df.columns:
        df["p_holm_report"] = df["p_holm"].apply(format_p_value)

    if "p_bonferroni" in df.columns:
        df["p_bonferroni_report"] = df["p_bonferroni"].apply(format_p_value)

    return df


def significance_label(p, alpha=0.05):
    if pd.isna(p):
        return "NA"
    return "Significant" if p < alpha else "Not Significant"


def maybe_sample(values, max_values=None, random_state=42):
    """
    Optional sampling only for visualization.
    It does not affect statistical tests.
    """
    values = np.asarray(values)

    if max_values is None:
        return values

    if len(values) <= max_values:
        return values

    rng = np.random.default_rng(random_state)
    sampled_idx = rng.choice(len(values), size=max_values, replace=False)
    return values[sampled_idx]


# ============================================================
# Data collector
# ============================================================

def collect_values_memory_safe():
    """
    Read all per-project CSV files chunk-by-chunk.

    This collector:
    - uses only Java files
    - uses all available Introduction values
    - uses all available Payment values
    - does not require files to be paired across phases for Mann-Whitney
    - computes delta only when both Introduction and Payment values exist
    - optionally filters indicator, e.g., VioMod only
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

    for metric_info in METRICS.values():
        required_cols.append(metric_info["intro_col"])
        required_cols.append(metric_info["payment_col"])

    required_cols = list(dict.fromkeys(required_cols))

    values = {}
    for metric_name in METRICS.keys():
        values[metric_name] = {}
        for group_name in GROUPS:
            values[metric_name][group_name] = {
                "intro": [],
                "payment": [],
                "delta": [],
            }

    file_count_summary = []

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        print(f"\nProcessing: {filename}")

        chunk_counter = 0
        total_rows_scanned = 0
        java_rows_detected = 0
        rows_after_indicator_filter = 0
        rows_after_group_filter = 0

        for chunk in pd.read_csv(
            csv_path,
            usecols=lambda c: c in required_cols,
            chunksize=CHUNKSIZE,
            low_memory=False,
        ):
            chunk_counter += 1
            total_rows_scanned += len(chunk)

            missing_cols = [c for c in required_cols if c not in chunk.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {filename}: {missing_cols}")

            # Keep only Java files
            chunk = chunk[
                chunk["file_path"]
                .astype(str)
                .str.lower()
                .str.endswith(".java", na=False)
            ].copy()

            if chunk.empty:
                continue

            java_rows_detected += len(chunk)

            # Optional indicator filter
            if FILTER_INDICATOR is not None:
                chunk = chunk[
                    chunk["indicator"].astype(str).str.lower().str.strip()
                    == FILTER_INDICATOR.lower()
                ].copy()

            if chunk.empty:
                continue

            rows_after_indicator_filter += len(chunk)

            # Normalize self-fixed group
            chunk["self_fixed_group"] = normalize_self_fixed_series(chunk["self_fixed"])
            chunk = chunk[chunk["self_fixed_group"].isin(GROUPS)].copy()

            if chunk.empty:
                continue

            rows_after_group_filter += len(chunk)

            # Collect metric values
            for metric_name, metric_info in METRICS.items():
                intro_col = metric_info["intro_col"]
                payment_col = metric_info["payment_col"]

                chunk[intro_col] = pd.to_numeric(chunk[intro_col], errors="coerce")
                chunk[payment_col] = pd.to_numeric(chunk[payment_col], errors="coerce")

                for group_name in GROUPS:
                    g = chunk[chunk["self_fixed_group"] == group_name]

                    if g.empty:
                        continue

                    intro_values = g[intro_col].dropna().to_numpy(dtype=float)
                    payment_values = g[payment_col].dropna().to_numpy(dtype=float)

                    values[metric_name][group_name]["intro"].append(intro_values)
                    values[metric_name][group_name]["payment"].append(payment_values)

                    complete_g = g.dropna(subset=[intro_col, payment_col])

                    if not complete_g.empty:
                        delta_values = (
                            complete_g[payment_col] - complete_g[intro_col]
                        ).to_numpy(dtype=float)

                        values[metric_name][group_name]["delta"].append(delta_values)

        file_count_summary.append({
            "source_file": filename,
            "chunks_read": chunk_counter,
            "total_rows_scanned": total_rows_scanned,
            "java_rows_detected_before_indicator_filter": java_rows_detected,
            "rows_after_indicator_filter": rows_after_indicator_filter,
            "rows_after_group_filter": rows_after_group_filter,
        })

        print(f"  Chunks read                         : {chunk_counter}")
        print(f"  Rows scanned                        : {total_rows_scanned}")
        print(f"  Java rows detected before filtering : {java_rows_detected}")
        print(f"  Rows after indicator filter         : {rows_after_indicator_filter}")
        print(f"  Rows after group filter             : {rows_after_group_filter}")

    compact_values = {}

    for metric_name in METRICS.keys():
        compact_values[metric_name] = {}

        for group_name in GROUPS:
            compact_values[metric_name][group_name] = {}

            for value_type in ["intro", "payment", "delta"]:
                arrays = values[metric_name][group_name][value_type]

                if len(arrays) == 0:
                    compact_values[metric_name][group_name][value_type] = np.array([])
                else:
                    compact_values[metric_name][group_name][value_type] = np.concatenate(arrays)

    file_count_summary_df = pd.DataFrame(file_count_summary)

    file_count_summary_path = os.path.join(
        OUTPUT_DIR,
        "RQ1_RQ2_java_file_rows_used_summary.csv"
    )

    file_count_summary_df.to_csv(file_count_summary_path, index=False)

    print("\nJava file row summary saved to:")
    print(file_count_summary_path)

    return compact_values, file_count_summary_df


# ============================================================
# Statistical result builders
# ============================================================

def build_descriptive_summary(values):
    rows = []

    for metric_name, metric_info in METRICS.items():
        rq = metric_info["rq"]

        for group_name in GROUPS:
            intro_values = values[metric_name][group_name]["intro"]
            payment_values = values[metric_name][group_name]["payment"]
            delta_values = values[metric_name][group_name]["delta"]

            rows.append({
                "RQ": rq,
                "Metric": metric_name,
                "Group": group_name,
                "N_Introduction": len(intro_values),
                "N_Payment": len(payment_values),
                "N_Delta_Complete_Intro_And_Payment": len(delta_values),
                "Introduction Median [IQR]": median_iqr(intro_values),
                "Payment Median [IQR]": median_iqr(payment_values),
                "Delta_Median": median_value(delta_values),
                "Delta Median [IQR]": median_iqr(delta_values),
                "Direction": direction_from_delta(delta_values),
            })

    return pd.DataFrame(rows)


def build_intro_vs_payment_tests(values):
    rows = []

    for metric_name, metric_info in METRICS.items():
        rq = metric_info["rq"]

        for group_name in GROUPS:
            intro_values = values[metric_name][group_name]["intro"]
            payment_values = values[metric_name][group_name]["payment"]
            delta_values = values[metric_name][group_name]["delta"]

            test_result = mann_whitney_result(
                intro_values,
                payment_values,
                x_label="Introduction",
                y_label="Payment",
            )

            rows.append({
                "RQ": rq,
                "Metric": metric_name,
                "Comparison": "Introduction vs Payment",
                "Group": group_name,
                "N_Introduction": len(intro_values),
                "N_Payment": len(payment_values),
                "N_Delta_Complete_Intro_And_Payment": len(delta_values),
                "Introduction Median [IQR]": median_iqr(intro_values),
                "Payment Median [IQR]": median_iqr(payment_values),
                "Delta_Median": median_value(delta_values),
                "Delta Median [IQR]": median_iqr(delta_values),
                **test_result,
            })

    result_df = pd.DataFrame(rows)

    result_df = adjust_p_values(result_df, p_col="p_value")
    result_df = add_p_value_report_columns(result_df)

    result_df["Significance_raw_p_0.05"] = result_df["p_value"].apply(significance_label)
    result_df["Significance_holm_0.05"] = result_df["p_holm"].apply(significance_label)
    result_df["Significance_bonferroni_0.05"] = result_df["p_bonferroni"].apply(significance_label)

    return result_df


def build_delta_group_tests(values):
    rows = []

    for metric_name, metric_info in METRICS.items():
        rq = metric_info["rq"]

        self_delta = values[metric_name]["Self-Fixed"]["delta"]
        non_self_delta = values[metric_name]["Non-Self-Fixed"]["delta"]

        test_result = mann_whitney_result(
            self_delta,
            non_self_delta,
            x_label="Self-Fixed Delta",
            y_label="Non-Self-Fixed Delta",
        )

        rows.append({
            "RQ": rq,
            "Metric": metric_name,
            "Comparison": "Delta Self-Fixed vs Delta Non-Self-Fixed",
            "N_Self_Fixed_Delta": len(self_delta),
            "N_Non_Self_Fixed_Delta": len(non_self_delta),
            "Self-Fixed Delta_Median": median_value(self_delta),
            "Non-Self-Fixed Delta_Median": median_value(non_self_delta),
            "Self-Fixed Delta Median [IQR]": median_iqr(self_delta),
            "Non-Self-Fixed Delta Median [IQR]": median_iqr(non_self_delta),
            **test_result,
        })

    result_df = pd.DataFrame(rows)

    result_df = adjust_p_values(result_df, p_col="p_value")
    result_df = add_p_value_report_columns(result_df)

    result_df["Significance_raw_p_0.05"] = result_df["p_value"].apply(significance_label)
    result_df["Significance_holm_0.05"] = result_df["p_holm"].apply(significance_label)
    result_df["Significance_bonferroni_0.05"] = result_df["p_bonferroni"].apply(significance_label)

    return result_df


def build_boxplot_descriptive_summary(values):
    rows = []

    for metric_name in METRICS.keys():
        for group_name in GROUPS:
            for phase_key, phase_label in [
                ("intro", "Introduction"),
                ("payment", "Payment")
            ]:
                arr = values[metric_name][group_name][phase_key]
                desc = descriptive_numeric(arr)

                rows.append({
                    "RQ": METRICS[metric_name]["rq"],
                    "Metric": metric_name,
                    "Group": group_name,
                    "Phase": phase_label,
                    **desc,
                })

            delta_arr = values[metric_name][group_name]["delta"]
            delta_desc = descriptive_numeric(delta_arr)

            rows.append({
                "RQ": METRICS[metric_name]["rq"],
                "Metric": metric_name,
                "Group": group_name,
                "Phase": "Delta Payment - Introduction",
                **delta_desc,
            })

    return pd.DataFrame(rows)


# ============================================================
# Plotting functions
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


def get_values_for_boxplot(values, metric_name, group_name, phase_key):
    arr = values[metric_name][group_name][phase_key]
    arr = maybe_sample(arr, max_values=MAX_VALUES_PER_BOX, random_state=42)
    return arr


def build_group_box_data(values, group_name, metric_names):
    phase_order = [
        (metric_names[0], "intro", "Introduction"),
        (metric_names[0], "payment", "Payment"),
        (metric_names[1], "intro", "Introduction"),
        (metric_names[1], "payment", "Payment"),
    ]

    box_values = []
    categories = []
    x_labels = []

    for metric_name, phase_key, phase_label in phase_order:
        arr = get_values_for_boxplot(
            values=values,
            metric_name=metric_name,
            group_name=group_name,
            phase_key=phase_key,
        )

        box_values.append(arr)
        categories.append(f"{metric_name} {phase_label}")

        metric_short = (
            metric_name
            .replace("Cyclomatic Complexity", "Cyclomatic")
            .replace("Cognitive Complexity", "Cognitive")
        )

        x_labels.append(f"{metric_short}\n{phase_label}")

    return box_values, categories, x_labels


def save_figure_png_and_pdf(fig, output_filename_without_ext):
    png_path = os.path.join(OUTPUT_DIR, f"{output_filename_without_ext}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{output_filename_without_ext}.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print("Saved figure:", png_path)
    print("Saved figure:", pdf_path)


def make_side_by_side_group_boxplot(
    values,
    metric_names,
    output_filename_without_ext,
    figure_title,
    left_group="Self-Fixed",
    right_group="Non-Self-Fixed",
):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6.5), sharey=False)

    group_specs = [
        (axes[0], left_group),
        (axes[1], right_group),
    ]

    for ax, group_name in group_specs:
        box_values, categories, x_labels = build_group_box_data(
            values=values,
            group_name=group_name,
            metric_names=metric_names,
        )

        box = ax.boxplot(
            box_values,
            labels=x_labels,
            showfliers=False,
            patch_artist=True,
            widths=0.6,
        )

        apply_box_colors(box, categories)

        ax.axvline(
            x=2.5,
            color=SEPARATOR_COLOR,
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
        )

        ax.set_title(group_name, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Metric and Phase", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(axis="y", linestyle="--", linewidth=0.8, color=GRID_COLOR, alpha=0.8)

    axes[0].set_ylabel("Metric Value", fontsize=12)

    intro_patch = plt.Line2D(
        [0], [0],
        marker="s",
        color="w",
        markerfacecolor=PHASE_COLORS["Introduction"],
        markeredgecolor=EDGE_COLOR,
        markersize=10,
        label="Introduction",
    )

    payment_patch = plt.Line2D(
        [0], [0],
        marker="s",
        color="w",
        markerfacecolor=PHASE_COLORS["Payment"],
        markeredgecolor=EDGE_COLOR,
        markersize=10,
        label="Payment",
    )

    fig.suptitle(figure_title, fontsize=16, fontweight="bold", y=0.985)

    fig.legend(
        handles=[intro_patch, payment_patch],
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.935),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.86])

    save_figure_png_and_pdf(fig, output_filename_without_ext)
    plt.close(fig)


def create_all_boxplots(values):
    make_side_by_side_group_boxplot(
        values=values,
        metric_names=["Fan-In", "Fan-Out"],
        output_filename_without_ext="RQ1_dependency_metrics_side_by_side_boxplot_file_level",
        figure_title="RQ1: Dependency Metrics (Self-Fixed vs Non-Self-Fixed)",
    )

    make_side_by_side_group_boxplot(
        values=values,
        metric_names=["Cyclomatic Complexity", "Cognitive Complexity"],
        output_filename_without_ext="RQ2_complexity_metrics_side_by_side_boxplot_file_level",
        figure_title="RQ2: Complexity Metrics (Self-Fixed vs Non-Self-Fixed)",
    )


# ============================================================
# Main
# ============================================================

def main():
    print("Starting combined memory-safe RQ1/RQ2 Mann-Whitney and boxplot analysis...")
    print("Input folder :", INPUT_DIR)
    print("Output folder:", OUTPUT_DIR)
    print("CSV pattern  :", CSV_PATTERN)
    print("Chunksize    :", CHUNKSIZE)
    print("Indicator    :", FILTER_INDICATOR)

    values, file_count_summary_df = collect_values_memory_safe()

    print("\nBuilding descriptive summary...")
    descriptive_df = build_descriptive_summary(values)

    print("\nRunning Mann-Whitney U tests: Introduction vs Payment...")
    intro_payment_df = build_intro_vs_payment_tests(values)

    print("\nRunning Mann-Whitney U tests: Delta Self-Fixed vs Delta Non-Self-Fixed...")
    delta_group_df = build_delta_group_tests(values)

    print("\nBuilding boxplot descriptive summary...")
    boxplot_summary_df = build_boxplot_descriptive_summary(values)

    descriptive_output = os.path.join(
        OUTPUT_DIR,
        "RQ1_RQ2_descriptive_summary_memory_safe.csv",
    )

    intro_payment_output = os.path.join(
        OUTPUT_DIR,
        "RQ1_RQ2_mannwhitney_intro_vs_payment_memory_safe.csv",
    )

    delta_group_output = os.path.join(
        OUTPUT_DIR,
        "RQ1_RQ2_mannwhitney_delta_self_vs_nonself_memory_safe.csv",
    )

    boxplot_summary_output = os.path.join(
        OUTPUT_DIR,
        "RQ1_RQ2_boxplot_descriptive_summary.csv",
    )

    descriptive_df.to_csv(
        descriptive_output,
        index=False,
        float_format="%.10e"
    )

    intro_payment_df.to_csv(
        intro_payment_output,
        index=False,
        float_format="%.10e"
    )

    delta_group_df.to_csv(
        delta_group_output,
        index=False,
        float_format="%.10e"
    )

    boxplot_summary_df.to_csv(
        boxplot_summary_output,
        index=False,
        float_format="%.10e"
    )

    print("\nCreating boxplots...")
    create_all_boxplots(values)

    print("\nDone.")
    print("Output folder:", OUTPUT_DIR)

    print("\nSaved CSV files:")
    print(descriptive_output)
    print(intro_payment_output)
    print(delta_group_output)
    print(boxplot_summary_output)
    print(os.path.join(OUTPUT_DIR, "RQ1_RQ2_java_file_rows_used_summary.csv"))

    print("\nDescriptive summary:")
    print(descriptive_df)

    print("\nIntroduction vs Payment Mann-Whitney tests:")
    print(intro_payment_df)

    print("\nDelta Self-Fixed vs Non-Self-Fixed Mann-Whitney tests:")
    print(delta_group_df)

    print("\nBoxplot descriptive summary:")
    print(boxplot_summary_df)

    print("\nFile row summary:")
    print(file_count_summary_df)


if __name__ == "__main__":
    main()