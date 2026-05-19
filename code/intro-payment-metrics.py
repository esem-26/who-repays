# 2
import os
import json
import glob
import pandas as pd


# ============================================================
# Input / Output
# ============================================================

INPUT_DIR = "ATD-DATA/SELF-FIXED/METRICS/FINAL"

OUTPUT_DIR = "ATD-DATA/SELF-FIXED/METRICS/FINAL/PER-BARIS"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATTERN = "*.csv"


# ============================================================
# Main columns
# ============================================================

ID_COLUMNS = [
    "No",
    "Key",
    "indicator",
    "Is Self-Fixed (Intro=Payment)",
    "Intro_Affected_Files",
]

INTRO_JSON_COL = "Intro_FileDetails_JSON"
PAYMENT_JSON_COL = "Payment_FileDetails_JSON"

METRICS = [
    "file_fan_in",
    "file_fan_out",
    "file_cyclomatic_complexity_sum",
    "file_cognitive_complexity_sum",
]


# ============================================================
# Helper functions
# ============================================================

def safe_parse_json(value):
    """
    Safely parse JSON string from CSV cell.

    Expected format:
    [
        {
            "file_path": "...",
            "file_fan_in": ...,
            "file_fan_out": ...,
            "file_cyclomatic_complexity_sum": ...,
            "file_cognitive_complexity_sum": ...
        }
    ]
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    if not isinstance(value, str):
        return []

    value = value.strip()

    if value == "" or value.lower() in ["nan", "none", "null"]:
        return []

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
        return []
    except json.JSONDecodeError:
        return []


def normalize_file_path(file_item):
    """
    Get file path from JSON item.
    Prefer file_path, fallback to other possible file-path keys.
    """
    if not isinstance(file_item, dict):
        return None

    return (
        file_item.get("file_path")
        or file_item.get("understand_file_relname")
        or file_item.get("path")
        or file_item.get("filename")
        or file_item.get("file")
    )


def build_file_metric_dict(file_details, prefix):
    """
    Convert list of file-level JSON objects into dictionary.

    Example output:
    {
        "path/FileA.java": {
            "intro_file_fan_in": 10,
            "intro_file_fan_out": 20,
            ...
        }
    }
    """
    result = {}

    for file_item in file_details:
        if not isinstance(file_item, dict):
            continue

        file_path = normalize_file_path(file_item)

        if file_path is None:
            continue

        metric_values = {}

        for metric in METRICS:
            metric_values[f"{prefix}_{metric}"] = file_item.get(metric)

        result[file_path] = metric_values

    return result


def convert_to_file_level(df, source_file):
    """
    Convert each ATD item into file-level rows.

    One output row represents one unique file related to one ATD item.
    The row contains Introduction and Payment metrics side by side.
    """
    output_rows = []

    for _, row in df.iterrows():

        base_info = {
            "source_file": source_file
        }

        for col in ID_COLUMNS:
            base_info[col] = row[col] if col in df.columns else None

        intro_details = safe_parse_json(row.get(INTRO_JSON_COL))
        payment_details = safe_parse_json(row.get(PAYMENT_JSON_COL))

        intro_dict = build_file_metric_dict(
            intro_details,
            prefix="intro"
        )

        payment_dict = build_file_metric_dict(
            payment_details,
            prefix="payment"
        )

        all_file_paths = sorted(
            set(intro_dict.keys()) | set(payment_dict.keys())
        )

        for file_path in all_file_paths:

            output_row = {
                **base_info,
                "file_path": file_path,
                "exists_in_intro": file_path in intro_dict,
                "exists_in_payment": file_path in payment_dict,
            }

            for metric in METRICS:
                intro_col = f"intro_{metric}"
                payment_col = f"payment_{metric}"

                output_row[intro_col] = intro_dict.get(file_path, {}).get(intro_col)
                output_row[payment_col] = payment_dict.get(file_path, {}).get(payment_col)

            output_rows.append(output_row)

    return pd.DataFrame(output_rows)


def add_numeric_and_delta_columns(file_level_df):
    """
    Convert metric columns to numeric and add delta columns.
    Delta = payment - introduction.
    """
    metric_columns = []

    for metric in METRICS:
        metric_columns.append(f"intro_{metric}")
        metric_columns.append(f"payment_{metric}")

    for col in metric_columns:
        if col in file_level_df.columns:
            file_level_df[col] = pd.to_numeric(file_level_df[col], errors="coerce")

    file_level_df["delta_file_fan_in"] = (
        file_level_df["payment_file_fan_in"]
        - file_level_df["intro_file_fan_in"]
    )

    file_level_df["delta_file_fan_out"] = (
        file_level_df["payment_file_fan_out"]
        - file_level_df["intro_file_fan_out"]
    )

    file_level_df["delta_file_cyclomatic_complexity_sum"] = (
        file_level_df["payment_file_cyclomatic_complexity_sum"]
        - file_level_df["intro_file_cyclomatic_complexity_sum"]
    )

    file_level_df["delta_file_cognitive_complexity_sum"] = (
        file_level_df["payment_file_cognitive_complexity_sum"]
        - file_level_df["intro_file_cognitive_complexity_sum"]
    )

    return file_level_df


def clean_and_sort(file_level_df):
    """
    Clean column names, filter Java files, and sort output.
    """

    # Rename self-fixed column
    if "Is Self-Fixed (Intro=Payment)" in file_level_df.columns:
        file_level_df = file_level_df.rename(
            columns={
                "Is Self-Fixed (Intro=Payment)": "self_fixed"
            }
        )

    # Keep only Java files
    if "file_path" in file_level_df.columns:
        file_level_df = file_level_df[
            file_level_df["file_path"].astype(str).str.endswith(".java", na=False)
        ].copy()

    # Sort output
    sort_columns = [
        "source_file",
        "Key",
        "file_path",
    ]

    existing_sort_columns = [
        col for col in sort_columns if col in file_level_df.columns
    ]

    if existing_sort_columns:
        file_level_df = file_level_df.sort_values(
            by=existing_sort_columns
        ).reset_index(drop=True)

    return file_level_df


# ============================================================
# Main process
# ============================================================

def main():

    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, CSV_PATTERN)))

    # Avoid reading output files again if the output folder is inside INPUT_DIR
    csv_files = [
        path for path in csv_files
        if os.path.abspath(OUTPUT_DIR) not in os.path.abspath(path)
    ]

    print(f"Found CSV files: {len(csv_files)}")

    if len(csv_files) == 0:
        print("No CSV files found.")
        return

    all_outputs = []

    for csv_path in csv_files:
        source_file = os.path.basename(csv_path)

        print("\nProcessing:", source_file)

        try:
            df = pd.read_csv(csv_path)

            file_level_df = convert_to_file_level(
                df=df,
                source_file=source_file
            )

            if file_level_df.empty:
                print("  Warning: no file-level rows generated.")
                continue

            file_level_df = add_numeric_and_delta_columns(file_level_df)
            file_level_df = clean_and_sort(file_level_df)

            output_filename = source_file.replace(".csv", "_PER_BARIS.csv")
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            file_level_df.to_csv(output_path, index=False)

            print(f"  Input rows : {len(df)}")
            print(f"  Output rows: {len(file_level_df)}")
            print(f"  Saved to   : {output_path}")

            all_outputs.append(file_level_df)

        except Exception as e:
            print(f"  ERROR processing {source_file}: {e}")

    # ========================================================
    # Save combined output from all CSV files
    # ========================================================

    if all_outputs:
        combined_df = pd.concat(all_outputs, ignore_index=True)

        combined_output_path = os.path.join(
            OUTPUT_DIR,
            "ALL_PROJECTS_FILE_LEVEL_PER_BARIS.csv"
        )

        combined_df.to_csv(combined_output_path, index=False)

        print("\nDone.")
        print(f"Total output rows: {len(combined_df)}")
        print(f"Combined file saved to: {combined_output_path}")

        print("\nPreview combined output:")
        print(combined_df.head(20))

    else:
        print("\nNo output generated.")


if __name__ == "__main__":
    main()