import os
import re
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


# ============================================================
# Input / Output Configuration
# ============================================================

INPUT_DIR = "ATD-DATA/SELF-FIXED/METRICS/FINAL"

CSV_PATTERN = "*UNDERSTAND-METRICS-CC-COMPLEXITY-FINAL.csv"

REPOS_ROOT = "repos"

OUTPUT_DIR = "ATD-DATA/SELF-FIXED/METRICS/FINAL/RQ3_FILE_LEVEL_CHANGE_ANALYSIS_VIOMOD"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Filter Configuration
# ============================================================

FILTER_INDICATOR = "viomod"


# ============================================================
# Repository Configuration
# ============================================================

AUTO_CLONE_REPOSITORY = True
GITHUB_ORG_DEFAULT = "apache"

FETCHED_REPOSITORIES: Set[str] = set()


# ============================================================
# Output Files
# ============================================================

OUTPUT_FILE_LEVEL_CSV = os.path.join(
    OUTPUT_DIR,
    "rq3_file_level_commit_count_with_metrics_viomod_only_FAST_9PROJECTS_CASSANDRA_FALLBACK.csv"
)

OUTPUT_SPEARMAN_CSV = os.path.join(
    OUTPUT_DIR,
    "rq3_spearman_commit_count_vs_metric_deltas_viomod_only_FAST_9PROJECTS_CASSANDRA_FALLBACK.csv"
)

OUTPUT_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "rq3_descriptive_summary_by_group_viomod_only_FAST_9PROJECTS_CASSANDRA_FALLBACK.csv"
)

OUTPUT_REPO_SELECTION_LOG_CSV = os.path.join(
    OUTPUT_DIR,
    "rq3_repository_selection_log_FAST_9PROJECTS_CASSANDRA_FALLBACK.csv"
)

OUTPUT_DEPENDENCY_SCATTER_PDF = os.path.join(
    OUTPUT_DIR,
    "rq3_dependency_file_commit_count_vs_metric_deltas.pdf"
)

OUTPUT_DEPENDENCY_SCATTER_PNG = os.path.join(
    OUTPUT_DIR,
    "rq3_dependency_file_commit_count_vs_metric_deltas.png"
)

OUTPUT_COMPLEXITY_SCATTER_PDF = os.path.join(
    OUTPUT_DIR,
    "rq3_complexity_file_commit_count_vs_metric_deltas.pdf"
)

OUTPUT_COMPLEXITY_SCATTER_PNG = os.path.join(
    OUTPUT_DIR,
    "rq3_complexity_file_commit_count_vs_metric_deltas.png"
)


# ============================================================
# Plot Configuration
# ============================================================

PLOT_DPI = 300
PLOT_MARKER_SIZE = 22
PLOT_ALPHA = 0.65
PLOT_SHOW_GRID = True

GROUP_PLOT_LABELS = {
    "Self-Fixed": "Self-fixed",
    "Non-Self-Fixed": "Non-self-fixed",
}

GROUP_COLORS = {
    "Self-Fixed": "#1f77b4",
    "Non-Self-Fixed": "#ff7f0e",
}


# ============================================================
# Repository Mapping
# ============================================================

PROJECT_REPO_FOLDER_OVERRIDES = {
    "https://github.com/apache/drill": "drill",
    "https://github.com/apache/camel": "camel",
    "https://github.com/apache/kafka": "kafka",
    "https://github.com/apache/cassandra": "cassandra",
    "https://github.com/apache/cassandra-analytics": "cassandra-analytics",
    "https://github.com/apache/activemq": "activemq",
    "https://github.com/apache/lucene": "lucene",
    "https://github.com/apache/solr": "solr",
    "https://github.com/apache/geode": "geode",
    "https://github.com/apache/netbeans": "netbeans",
}

PROJECT_KEY_TO_REPO_FOLDER = {
    "DRILL": "drill",
    "CAMEL": "camel",
    "KAFKA": "kafka",
    "CASSANDRA": "cassandra",
    "AMQ": "activemq",
    "LUCENE": "lucene",
    "SOLR": "solr",
    "GEODE": "geode",
    "NETBEANS": "netbeans",
}

ALLOWED_REPO_FOLDERS = {
    "drill",
    "camel",
    "kafka",
    "cassandra",
    "cassandra-analytics",
    "activemq",
    "lucene",
    "solr",
    "geode",
    "netbeans",
}

PROJECT_REPO_CANDIDATES = {
    "CASSANDRA": [
        {
            "repo_folder": "cassandra",
            "clone_url": "https://github.com/apache/cassandra.git",
        },
        {
            "repo_folder": "cassandra-analytics",
            "clone_url": "https://github.com/apache/cassandra-analytics.git",
        },
    ]
}


# ============================================================
# Basic Utilities
# ============================================================

def normalize_path(path: Any) -> Optional[str]:
    if path is None or pd.isna(path):
        return None

    path = str(path).strip()
    path = path.strip('"').strip("'")
    path = path.replace("\\", "/")
    path = re.sub(r"/+", "/", path)

    if path.startswith("./"):
        path = path[2:]

    return path


def is_java_file(path: Any) -> bool:
    path = normalize_path(path)

    if not path:
        return False

    return path.lower().endswith(".java")


def safe_json_loads(value: Any, default: Any = None) -> Any:
    if default is None:
        default = []

    if value is None or pd.isna(value):
        return default

    if isinstance(value, (dict, list)):
        return value

    try:
        return json.loads(str(value))
    except Exception:
        return default


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def parse_intro_affected_files(value: Any) -> List[str]:
    if value is None or pd.isna(value):
        return []

    files = []

    for item in re.split(r";|\n|\r", str(value)):
        item = normalize_path(item)

        if item and is_java_file(item):
            files.append(item)

    return sorted(set(files))


def delta_value(payment_value: Any, intro_value: Any) -> float:
    payment_value = pd.to_numeric(payment_value, errors="coerce")
    intro_value = pd.to_numeric(intro_value, errors="coerce")

    if pd.isna(payment_value) or pd.isna(intro_value):
        return np.nan

    return float(payment_value - intro_value)


# ============================================================
# Load Input CSVs
# ============================================================

def load_all_input_csvs(input_dir: str, csv_pattern: str) -> pd.DataFrame:
    input_dir = Path(input_dir)
    csv_files = sorted(input_dir.glob(csv_pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in:\n{input_dir}\nwith pattern:\n{csv_pattern}"
        )

    print("============================================================")
    print("Input CSV Files")
    print("============================================================")

    dfs = []

    for csv_file in csv_files:
        print(f"Loading: {csv_file}")
        temp_df = pd.read_csv(csv_file)
        temp_df["source_csv_file"] = csv_file.name
        dfs.append(temp_df)

    combined_df = pd.concat(dfs, ignore_index=True)

    print("============================================================")
    print(f"Total CSV files loaded : {len(csv_files)}")
    print(f"Total rows combined    : {len(combined_df)}")
    print("============================================================\n")

    return combined_df


# ============================================================
# Git Utilities
# ============================================================

def run_git(repo_dir: Path, args: List[str], allow_error: bool = False) -> str:
    cmd = ["git", "-C", str(repo_dir)] + args

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0 and not allow_error:
        raise RuntimeError(
            f"Git command failed:\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout.strip()


def fetch_all_branches_and_tags(repo_dir: Path) -> None:
    repo_key = str(repo_dir)

    if repo_key in FETCHED_REPOSITORIES:
        return

    print(f"[FETCH] Updating repository once: {repo_dir}")

    subprocess.run(
        [
            "git",
            "-C",
            str(repo_dir),
            "fetch",
            "--all",
            "--tags",
            "--prune"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    FETCHED_REPOSITORIES.add(repo_key)


def commit_exists(repo_dir: Path, commit_hash: Any) -> bool:
    if commit_hash is None or pd.isna(commit_hash):
        return False

    commit_hash = str(commit_hash).strip()

    if not commit_hash:
        return False

    def _cat_file_exists() -> bool:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_dir),
                "cat-file",
                "-e",
                f"{commit_hash}^{{commit}}"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode == 0

    if _cat_file_exists():
        return True

    fetch_all_branches_and_tags(repo_dir)

    if _cat_file_exists():
        return True

    print(f"[FETCH-SHA] Commit not local. Trying exact fetch: {commit_hash}")

    fetch_attempts = [
        ["fetch", "origin", commit_hash],
        ["fetch", "origin", commit_hash, "--depth=1"],
        ["fetch", "origin", f"+{commit_hash}:refs/temp/commit-{commit_hash[:12]}"],
    ]

    for args in fetch_attempts:
        subprocess.run(
            ["git", "-C", str(repo_dir)] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if _cat_file_exists():
            print(f"[OK] Commit fetched successfully: {commit_hash}")
            return True

    print(f"[MISS] Commit still not found after exact fetch: {commit_hash}")

    return False


def get_parent_commit(repo_dir: Path, commit_hash: str) -> Optional[str]:
    output = run_git(
        repo_dir,
        ["rev-list", "--parents", "-n", "1", str(commit_hash).strip()],
        allow_error=True
    )

    parts = output.split()

    if len(parts) >= 2:
        return parts[1]

    return None


def build_commit_range(
    repo_dir: Path,
    intro_hash: str,
    payment_hash: str,
    include_intro: bool = True
) -> str:
    intro_hash = str(intro_hash).strip()
    payment_hash = str(payment_hash).strip()

    if include_intro:
        parent = get_parent_commit(repo_dir, intro_hash)

        if parent:
            return f"{parent}..{payment_hash}"

        return payment_hash

    return f"{intro_hash}..{payment_hash}"


def get_files_at_commit(repo_dir: Path, commit_hash: str) -> Set[str]:
    output = run_git(
        repo_dir,
        ["ls-tree", "-r", "--name-only", str(commit_hash).strip()],
        allow_error=True
    )

    return {
        normalize_path(line)
        for line in output.splitlines()
        if normalize_path(line)
    }


# ============================================================
# Fast Git Log Parsing
# ============================================================

def parse_git_log_name_status(output: str) -> List[Dict[str, Any]]:
    records = []
    current_record = None

    for raw_line in output.splitlines():
        line = raw_line.rstrip("\n")

        if not line:
            continue

        if line.startswith("__COMMIT__"):
            commit_hash = line.replace("__COMMIT__", "").strip()

            current_record = {
                "commit": commit_hash,
                "changes": []
            }

            records.append(current_record)
            continue

        if current_record is None:
            continue

        parts = line.split("\t")

        if not parts:
            continue

        status = parts[0].strip()

        if status.startswith("R") or status.startswith("C"):
            if len(parts) >= 3:
                current_record["changes"].append({
                    "status": status,
                    "old_path": normalize_path(parts[1]),
                    "new_path": normalize_path(parts[2]),
                })
        else:
            if len(parts) >= 2:
                path = normalize_path(parts[1])
                current_record["changes"].append({
                    "status": status,
                    "old_path": path,
                    "new_path": path,
                })

    return records


def get_issue_change_log(repo_dir: Path, commit_range: str) -> List[Dict[str, Any]]:
    output = run_git(
        repo_dir,
        [
            "log",
            "--reverse",
            "--name-status",
            "-M",
            "-C",
            "--format=__COMMIT__%H",
            commit_range
        ],
        allow_error=True
    )

    return parse_git_log_name_status(output)


def fast_track_files_for_issue(
    repo_dir: Path,
    intro_hash: str,
    payment_hash: str,
    intro_files: List[str],
    include_intro: bool = True
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    intro_files = sorted(
        set([
            normalize_path(p)
            for p in intro_files
            if is_java_file(p)
        ])
    )

    commit_range = build_commit_range(
        repo_dir=repo_dir,
        intro_hash=intro_hash,
        payment_hash=payment_hash,
        include_intro=include_intro
    )

    files_at_intro = get_files_at_commit(repo_dir, intro_hash)

    tracking: Dict[str, Dict[str, Any]] = {}

    for intro_file in intro_files:
        exists_at_intro = intro_file in files_at_intro

        tracking[intro_file] = {
            "intro_file": intro_file,
            "current_path": intro_file if exists_at_intro else None,
            "final_path": intro_file if exists_at_intro else None,
            "exists_at_intro": exists_at_intro,
            "deleted": False,
            "rename_count": 0,
            "touched_commits": [],
            "path_history": [intro_file],
            "tracking_status": (
                "tracked_to_payment"
                if exists_at_intro
                else "intro_file_not_found_at_intro_commit"
            ),
        }

    commit_records = get_issue_change_log(repo_dir, commit_range)

    for record in commit_records:
        commit_hash = record["commit"]
        changes = record["changes"]

        current_path_to_intro_files: Dict[str, List[str]] = {}

        for intro_file, state in tracking.items():
            if state["deleted"]:
                continue

            current_path = state["current_path"]

            if current_path:
                current_path_to_intro_files.setdefault(
                    current_path,
                    []
                ).append(intro_file)

        for change in changes:
            status = change["status"]
            old_path = change["old_path"]
            new_path = change["new_path"]

            if status.startswith("R"):
                if old_path in current_path_to_intro_files:
                    for intro_file in current_path_to_intro_files[old_path]:
                        state = tracking[intro_file]
                        state["touched_commits"].append(commit_hash)
                        state["current_path"] = new_path
                        state["final_path"] = new_path
                        state["rename_count"] += 1
                        state["path_history"].append(new_path)

            elif status.startswith("C"):
                if old_path in current_path_to_intro_files:
                    for intro_file in current_path_to_intro_files[old_path]:
                        state = tracking[intro_file]
                        state["touched_commits"].append(commit_hash)

            elif status == "D":
                if old_path in current_path_to_intro_files:
                    for intro_file in current_path_to_intro_files[old_path]:
                        state = tracking[intro_file]
                        state["touched_commits"].append(commit_hash)
                        state["deleted"] = True
                        state["final_path"] = None
                        state["current_path"] = None
                        state["tracking_status"] = "deleted_before_or_at_payment"
                        state["path_history"].append("[DELETED]")

            else:
                candidate_paths = set()

                if old_path:
                    candidate_paths.add(old_path)

                if new_path:
                    candidate_paths.add(new_path)

                for candidate_path in candidate_paths:
                    if candidate_path in current_path_to_intro_files:
                        for intro_file in current_path_to_intro_files[candidate_path]:
                            state = tracking[intro_file]
                            state["touched_commits"].append(commit_hash)

    for intro_file, state in tracking.items():
        unique_commits = list(dict.fromkeys(state["touched_commits"]))
        state["touched_commits"] = unique_commits
        state["commit_count"] = len(unique_commits)

        if state["tracking_status"] == "tracked_to_payment" and not state["deleted"]:
            state["final_path"] = state["current_path"]

    return tracking, commit_range


# ============================================================
# Repository Utilities
# ============================================================

def get_project_key_from_row(row: pd.Series) -> str:
    key = str(row.get("Key", "")).strip().upper()

    if "-" in key:
        return key.split("-")[0]

    github_url = str(row.get("GitHub Base URL", "")).strip().rstrip("/")

    if github_url and github_url.lower() != "nan":
        return github_url.split("/")[-1].replace(".git", "").upper()

    return ""


def get_repo_folder_from_row(row: pd.Series) -> str:
    github_url = str(row.get("GitHub Base URL", "")).strip().rstrip("/")

    if github_url in PROJECT_REPO_FOLDER_OVERRIDES:
        return PROJECT_REPO_FOLDER_OVERRIDES[github_url]

    key = str(row.get("Key", "")).strip().upper()

    if "-" in key:
        project_key = key.split("-")[0]

        if project_key in PROJECT_KEY_TO_REPO_FOLDER:
            return PROJECT_KEY_TO_REPO_FOLDER[project_key]

    raise ValueError(
        f"Cannot determine repository folder for row with "
        f"Key={row.get('Key')} and GitHub Base URL={github_url}"
    )


def get_clone_url_from_repo_folder(repo_folder: str) -> str:
    if repo_folder not in ALLOWED_REPO_FOLDERS:
        raise ValueError(
            f"Repository folder is not in the allowed repositories: {repo_folder}"
        )

    return f"https://github.com/apache/{repo_folder}.git"


def clone_repository_if_missing(repo_dir: Path, clone_url: str) -> bool:
    repo_dir = Path(repo_dir)

    if repo_dir.exists() and (repo_dir / ".git").exists():
        return True

    if repo_dir.exists() and not (repo_dir / ".git").exists():
        print(f"[WARNING] Folder exists but is not a Git repository: {repo_dir}")
        return False

    if not AUTO_CLONE_REPOSITORY:
        print(f"[SKIP] Repository not found and AUTO_CLONE_REPOSITORY=False: {repo_dir}")
        return False

    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("Repository Not Found. Cloning Repository")
    print("============================================================")
    print(f"Clone URL : {clone_url}")
    print(f"Target    : {repo_dir}")
    print("============================================================")

    result = subprocess.run(
        ["git", "clone", clone_url, str(repo_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(f"[ERROR] Failed to clone repository: {clone_url}")
        print(result.stderr)
        return False

    print(f"[OK] Repository cloned successfully: {repo_dir}")

    return True


def prepare_repository_for_row(
    row: pd.Series,
    intro_hash: str,
    payment_hash: str
) -> Dict[str, Any]:
    project_key = get_project_key_from_row(row)

    default_repo_folder = get_repo_folder_from_row(row)
    default_clone_url = get_clone_url_from_repo_folder(default_repo_folder)

    if project_key in PROJECT_REPO_CANDIDATES:
        candidates = PROJECT_REPO_CANDIDATES[project_key].copy()

        default_candidate = {
            "repo_folder": default_repo_folder,
            "clone_url": default_clone_url,
        }

        if default_candidate not in candidates:
            candidates.insert(0, default_candidate)
    else:
        candidates = [
            {
                "repo_folder": default_repo_folder,
                "clone_url": default_clone_url,
            }
        ]

    tried = []

    for candidate in candidates:
        repo_folder = candidate["repo_folder"]
        clone_url = candidate["clone_url"]

        if repo_folder not in ALLOWED_REPO_FOLDERS:
            tried.append({
                "project_key": project_key,
                "repo_folder": repo_folder,
                "clone_url": clone_url,
                "repo_dir": None,
                "repo_ready": False,
                "intro_commit_found": False,
                "payment_commit_found": False,
                "status": "repo_not_allowed",
            })
            continue

        repo_dir = Path(REPOS_ROOT) / repo_folder

        repo_ready = clone_repository_if_missing(
            repo_dir=repo_dir,
            clone_url=clone_url
        )

        if not repo_ready:
            tried.append({
                "project_key": project_key,
                "repo_folder": repo_folder,
                "clone_url": clone_url,
                "repo_dir": str(repo_dir),
                "repo_ready": False,
                "intro_commit_found": False,
                "payment_commit_found": False,
                "status": "repository_unavailable",
            })
            continue

        intro_exists = commit_exists(repo_dir, intro_hash)
        payment_exists = commit_exists(repo_dir, payment_hash)

        tried.append({
            "project_key": project_key,
            "repo_folder": repo_folder,
            "clone_url": clone_url,
            "repo_dir": str(repo_dir),
            "repo_ready": True,
            "intro_commit_found": intro_exists,
            "payment_commit_found": payment_exists,
            "status": (
                "matched_intro_and_payment"
                if intro_exists and payment_exists
                else "commit_not_found_in_this_candidate"
            ),
        })

        if intro_exists and payment_exists:
            return {
                "repo_ready": True,
                "project_key": project_key,
                "repo_folder": repo_folder,
                "repo_dir": repo_dir,
                "clone_url": clone_url,
                "tried_repositories": tried,
                "repo_selection_status": "matched_intro_and_payment",
                "intro_commit_found": True,
                "payment_commit_found": True,
            }

        print(
            f"[REPO-MISS] {project_key} | repo={repo_folder} | "
            f"intro_found={intro_exists} | payment_found={payment_exists}"
        )

    return {
        "repo_ready": False,
        "project_key": project_key,
        "repo_folder": default_repo_folder,
        "repo_dir": Path(REPOS_ROOT) / default_repo_folder,
        "clone_url": default_clone_url,
        "tried_repositories": tried,
        "repo_selection_status": "commit_not_found_in_expected_or_fallback_repository",
        "intro_commit_found": False,
        "payment_commit_found": False,
    }


def get_self_fixed_group(row: pd.Series) -> Tuple[bool, str]:
    value = str(row.get("Is Self-Fixed (Intro=Payment)", "")).strip().lower()

    is_self_fixed = value in ["true", "1", "yes", "y"]

    group = "Self-Fixed" if is_self_fixed else "Non-Self-Fixed"

    return is_self_fixed, group


# ============================================================
# Metric Parsing
# ============================================================

def build_metric_map(json_value: Any) -> Dict[str, Dict[str, float]]:
    details = safe_json_loads(json_value, default=[])

    if not isinstance(details, list):
        return {}

    metric_map = {}

    for item in details:
        if not isinstance(item, dict):
            continue

        file_path = normalize_path(
            item.get("file_path")
            or item.get("understand_file_relname")
            or item.get("git_numstat_path")
            or item.get("path")
            or item.get("file")
            or item.get("name")
        )

        if not is_java_file(file_path):
            continue

        metric_map[file_path] = {
            "file_fan_in": safe_float(
                item.get("file_fan_in", item.get("fan_in", np.nan))
            ),
            "file_fan_out": safe_float(
                item.get("file_fan_out", item.get("fan_out", np.nan))
            ),
            "file_cyclomatic_complexity_sum": safe_float(
                item.get(
                    "file_cyclomatic_complexity_sum",
                    item.get(
                        "cyclomatic_complexity_sum",
                        item.get(
                            "file_cyclomatic_complexity",
                            item.get("cyclomatic_complexity", np.nan)
                        )
                    )
                )
            ),
            "file_cognitive_complexity_sum": safe_float(
                item.get(
                    "file_cognitive_complexity_sum",
                    item.get(
                        "cognitive_complexity_sum",
                        item.get(
                            "file_cognitive_complexity",
                            item.get("cognitive_complexity", np.nan)
                        )
                    )
                )
            ),
        }

    return metric_map


def parse_payment_path_map(value: Any) -> Dict[str, Optional[str]]:
    data = safe_json_loads(value, default={})

    if not isinstance(data, dict):
        return {}

    result = {}

    for k, v in data.items():
        key = normalize_path(k)
        val = normalize_path(v) if v else None

        if key:
            result[key] = val

    return result


def get_payment_metric_for_intro_file_fast(
    intro_file: str,
    final_path: Optional[str],
    payment_metric_map: Dict[str, Dict[str, float]],
    payment_path_map: Dict[str, Optional[str]]
) -> Tuple[Optional[str], Dict[str, float]]:
    empty = {
        "file_fan_in": np.nan,
        "file_fan_out": np.nan,
        "file_cyclomatic_complexity_sum": np.nan,
        "file_cognitive_complexity_sum": np.nan,
    }

    candidate_paths = []

    if final_path:
        candidate_paths.append(final_path)

    mapped_path = payment_path_map.get(intro_file)

    if mapped_path:
        candidate_paths.append(mapped_path)

    candidate_paths.append(intro_file)

    candidate_paths = [
        normalize_path(p)
        for p in candidate_paths
        if normalize_path(p)
    ]

    candidate_paths = list(dict.fromkeys(candidate_paths))

    for path in candidate_paths:
        if path in payment_metric_map:
            return path, payment_metric_map[path]

    return None, empty


# ============================================================
# RQ3 File-Level Dataset Construction
# ============================================================

def build_file_level_rq3_dataset(
    input_dir: str,
    csv_pattern: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_all_input_csvs(input_dir, csv_pattern)

    required_columns = [
        "Key",
        "Intro Hash",
        "Payment Hash",
        "Intro_Affected_Files",
        "Is Self-Fixed (Intro=Payment)",
        "Intro_FileDetails_JSON",
        "Payment_FileDetails_JSON",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if FILTER_INDICATOR is not None:
        if "indicator" not in df.columns:
            raise ValueError(
                "FILTER_INDICATOR is enabled, but column 'indicator' is missing."
            )

        before_filter = len(df)

        df = df[
            df["indicator"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(str(FILTER_INDICATOR).lower())
        ].copy()

        after_filter = len(df)

        print("============================================================")
        print("Indicator Filter")
        print("============================================================")
        print(f"Filter indicator : {FILTER_INDICATOR}")
        print(f"Rows before      : {before_filter}")
        print(f"Rows after       : {after_filter}")
        print("============================================================")

    file_level_rows = []
    repo_selection_log_rows = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="RQ3 fast file-level tracking"
    ):
        key = row["Key"]
        intro_hash = str(row["Intro Hash"]).strip()
        payment_hash = str(row["Payment Hash"]).strip()

        try:
            repo_info = prepare_repository_for_row(
                row=row,
                intro_hash=intro_hash,
                payment_hash=payment_hash
            )
        except Exception as e:
            repo_info = {
                "repo_ready": False,
                "project_key": get_project_key_from_row(row),
                "repo_folder": None,
                "repo_dir": None,
                "clone_url": None,
                "tried_repositories": [],
                "repo_selection_status": f"repository_selection_error: {e}",
                "intro_commit_found": False,
                "payment_commit_found": False,
            }

        repo_selection_log_rows.append({
            "Key": key,
            "source_csv_file": row.get("source_csv_file", np.nan),
            "project_key": repo_info.get("project_key"),
            "repo_ready": repo_info.get("repo_ready"),
            "repo_folder": repo_info.get("repo_folder"),
            "repo_dir": str(repo_info.get("repo_dir")) if repo_info.get("repo_dir") else None,
            "clone_url": repo_info.get("clone_url"),
            "repo_selection_status": repo_info.get("repo_selection_status"),
            "intro_hash": intro_hash,
            "payment_hash": payment_hash,
            "intro_commit_found": repo_info.get("intro_commit_found"),
            "payment_commit_found": repo_info.get("payment_commit_found"),
            "tried_repositories_json": json.dumps(
                repo_info.get("tried_repositories", []),
                ensure_ascii=False
            ),
        })

        if not repo_info["repo_ready"]:
            print(
                f"[SKIP] Repository/commit unavailable for {key} | "
                f"repo_status={repo_info.get('repo_selection_status')}"
            )
            continue

        repo_folder = repo_info["repo_folder"]
        repo_dir = repo_info["repo_dir"]
        clone_url = repo_info["clone_url"]
        repo_selection_status = repo_info["repo_selection_status"]

        java_files = parse_intro_affected_files(row["Intro_Affected_Files"])

        if not java_files:
            print(f"[SKIP] No Java files in Intro_Affected_Files for {key}")
            continue

        intro_metric_map = build_metric_map(row["Intro_FileDetails_JSON"])
        payment_metric_map = build_metric_map(row["Payment_FileDetails_JSON"])

        payment_path_map = {}

        if "Payment_Path_Map_JSON" in df.columns:
            payment_path_map = parse_payment_path_map(
                row.get("Payment_Path_Map_JSON")
            )

        is_self_fixed, group = get_self_fixed_group(row)

        tracking_by_file, commit_range = fast_track_files_for_issue(
            repo_dir=repo_dir,
            intro_hash=intro_hash,
            payment_hash=payment_hash,
            intro_files=java_files,
            include_intro=True
        )

        for intro_file in java_files:
            tracking_result = tracking_by_file.get(intro_file)

            if tracking_result is None:
                continue

            intro_metrics = intro_metric_map.get(
                intro_file,
                {
                    "file_fan_in": np.nan,
                    "file_fan_out": np.nan,
                    "file_cyclomatic_complexity_sum": np.nan,
                    "file_cognitive_complexity_sum": np.nan,
                }
            )

            final_path = tracking_result.get("final_path")

            payment_metric_path, payment_metrics = get_payment_metric_for_intro_file_fast(
                intro_file=intro_file,
                final_path=final_path,
                payment_metric_map=payment_metric_map,
                payment_path_map=payment_path_map
            )

            intro_fan_in = safe_float(intro_metrics["file_fan_in"])
            payment_fan_in = safe_float(payment_metrics["file_fan_in"])

            intro_fan_out = safe_float(intro_metrics["file_fan_out"])
            payment_fan_out = safe_float(payment_metrics["file_fan_out"])

            intro_cyclomatic = safe_float(
                intro_metrics["file_cyclomatic_complexity_sum"]
            )
            payment_cyclomatic = safe_float(
                payment_metrics["file_cyclomatic_complexity_sum"]
            )

            intro_cognitive = safe_float(
                intro_metrics["file_cognitive_complexity_sum"]
            )
            payment_cognitive = safe_float(
                payment_metrics["file_cognitive_complexity_sum"]
            )

            touched_commits = tracking_result.get("touched_commits", [])

            file_level_rows.append({
                "source_csv_file": row.get("source_csv_file", np.nan),

                "Key": key,
                "label": row.get("label", np.nan),
                "indicator": row.get("indicator", np.nan),

                "group": group,
                "is_self_fixed": is_self_fixed,

                "project_key": repo_info.get("project_key"),
                "repo_folder": repo_folder,
                "clone_url": clone_url,
                "repo_selection_status": repo_selection_status,

                "intro_hash": intro_hash,
                "payment_hash": payment_hash,
                "commit_range_used": commit_range,

                "intro_file_path": intro_file,
                "exists_at_intro": tracking_result.get("exists_at_intro"),
                "payment_metric_path": payment_metric_path,
                "final_tracked_path": final_path,

                "tracking_status": tracking_result.get("tracking_status"),
                "deleted": tracking_result.get("deleted"),
                "rename_count": tracking_result.get("rename_count"),
                "path_history": ";".join(tracking_result.get("path_history", [])),

                "file_commit_count_intro_to_payment": tracking_result.get("commit_count", 0),
                "touched_commits": ";".join(touched_commits),

                "intro_fan_in": intro_fan_in,
                "payment_fan_in": payment_fan_in,
                "delta_fan_in": delta_value(payment_fan_in, intro_fan_in),

                "intro_fan_out": intro_fan_out,
                "payment_fan_out": payment_fan_out,
                "delta_fan_out": delta_value(payment_fan_out, intro_fan_out),

                "intro_cyclomatic_complexity": intro_cyclomatic,
                "payment_cyclomatic_complexity": payment_cyclomatic,
                "delta_cyclomatic_complexity": delta_value(
                    payment_cyclomatic,
                    intro_cyclomatic
                ),

                "intro_cognitive_complexity": intro_cognitive,
                "payment_cognitive_complexity": payment_cognitive,
                "delta_cognitive_complexity": delta_value(
                    payment_cognitive,
                    intro_cognitive
                ),
            })

    file_df = pd.DataFrame(file_level_rows)
    repo_selection_log_df = pd.DataFrame(repo_selection_log_rows)

    return file_df, repo_selection_log_df


# ============================================================
# RQ3 Statistical Analysis with Effect Size
# ============================================================

def interpret_correlation_effect_size(rho: float) -> str:
    if pd.isna(rho):
        return "NA"

    abs_rho = abs(rho)

    if abs_rho < 0.10:
        return "negligible"
    elif abs_rho < 0.30:
        return "weak"
    elif abs_rho < 0.50:
        return "moderate"
    else:
        return "strong"


def interpret_correlation_direction(rho: float) -> str:
    if pd.isna(rho):
        return "NA"

    if rho > 0:
        return "positive"
    elif rho < 0:
        return "negative"
    else:
        return "zero"


def spearman_analysis(file_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "delta_fan_in",
        "delta_fan_out",
        "delta_cyclomatic_complexity",
        "delta_cognitive_complexity",
    ]

    metric_labels = {
        "delta_fan_in": "Delta Fan-In",
        "delta_fan_out": "Delta Fan-Out",
        "delta_cyclomatic_complexity": "Delta Cyclomatic Complexity",
        "delta_cognitive_complexity": "Delta Cognitive Complexity",
    }

    groups = ["All"] + sorted(file_df["group"].dropna().unique().tolist())

    results = []

    for group in groups:
        if group == "All":
            subset = file_df.copy()
        else:
            subset = file_df[file_df["group"] == group].copy()

        for metric in metric_cols:
            temp = subset[
                [
                    "file_commit_count_intro_to_payment",
                    metric
                ]
            ].dropna()

            n = len(temp)

            if n < 3:
                rho = np.nan
                p_value = np.nan
                note = "Insufficient data"
            elif temp["file_commit_count_intro_to_payment"].nunique() < 2:
                rho = np.nan
                p_value = np.nan
                note = "Commit count has no variation"
            elif temp[metric].nunique() < 2:
                rho = np.nan
                p_value = np.nan
                note = "Metric delta has no variation"
            else:
                rho, p_value = spearmanr(
                    temp["file_commit_count_intro_to_payment"],
                    temp[metric]
                )
                note = "OK"

            results.append({
                "Group": group,
                "Metric_Delta": metric,
                "Metric_Label": metric_labels.get(metric, metric),
                "N": n,
                "Spearman_Rho": rho,
                "Effect_Size_Abs_Rho": abs(rho) if not pd.isna(rho) else np.nan,
                "Effect_Size_Category": interpret_correlation_effect_size(rho),
                "Correlation_Direction": interpret_correlation_direction(rho),
                "P_Value": p_value,
                "Significant_Raw_0.05": (
                    bool(p_value < 0.05)
                    if not pd.isna(p_value)
                    else False
                ),
                "Note": note,
            })

    result_df = pd.DataFrame(results)

    result_df["P_Value_Holm"] = np.nan
    result_df["P_Value_Bonferroni"] = np.nan
    result_df["Significant_Holm_0.05"] = False
    result_df["Significant_Bonferroni_0.05"] = False

    valid_mask = result_df["P_Value"].notna()

    if valid_mask.sum() > 0:
        pvals = result_df.loc[valid_mask, "P_Value"].values

        holm_reject, holm_pvals, _, _ = multipletests(
            pvals,
            alpha=0.05,
            method="holm"
        )

        bonf_reject, bonf_pvals, _, _ = multipletests(
            pvals,
            alpha=0.05,
            method="bonferroni"
        )

        result_df.loc[valid_mask, "P_Value_Holm"] = holm_pvals
        result_df.loc[valid_mask, "P_Value_Bonferroni"] = bonf_pvals
        result_df.loc[valid_mask, "Significant_Holm_0.05"] = holm_reject
        result_df.loc[valid_mask, "Significant_Bonferroni_0.05"] = bonf_reject

    ordered_cols = [
        "Group",
        "Metric_Delta",
        "Metric_Label",
        "N",
        "Spearman_Rho",
        "Effect_Size_Abs_Rho",
        "Effect_Size_Category",
        "Correlation_Direction",
        "P_Value",
        "Significant_Raw_0.05",
        "P_Value_Holm",
        "Significant_Holm_0.05",
        "P_Value_Bonferroni",
        "Significant_Bonferroni_0.05",
        "Note",
    ]

    return result_df[ordered_cols]


def descriptive_summary(file_df: pd.DataFrame) -> pd.DataFrame:
    summary_cols = [
        "file_commit_count_intro_to_payment",
        "delta_fan_in",
        "delta_fan_out",
        "delta_cyclomatic_complexity",
        "delta_cognitive_complexity",
    ]

    rows = []
    groups = ["All"] + sorted(file_df["group"].dropna().unique().tolist())

    for group in groups:
        if group == "All":
            subset = file_df.copy()
        else:
            subset = file_df[file_df["group"] == group].copy()

        for col in summary_cols:
            values = subset[col].dropna()

            if len(values) == 0:
                rows.append({
                    "Group": group,
                    "Variable": col,
                    "N": 0,
                    "Mean": np.nan,
                    "Median": np.nan,
                    "Q1": np.nan,
                    "Q3": np.nan,
                    "IQR": np.nan,
                    "Min": np.nan,
                    "Max": np.nan,
                })
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)

            rows.append({
                "Group": group,
                "Variable": col,
                "N": len(values),
                "Mean": values.mean(),
                "Median": values.median(),
                "Q1": q1,
                "Q3": q3,
                "IQR": q3 - q1,
                "Min": values.min(),
                "Max": values.max(),
            })

    return pd.DataFrame(rows)


# ============================================================
# Scatter Plot: Dependency and Complexity Figures
# ============================================================

def add_regression_line(ax, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if len(x) < 2:
        return

    if len(np.unique(x)) < 2:
        return

    try:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        y_line = slope * x_line + intercept

        ax.plot(
            x_line,
            y_line,
            color=color,
            linewidth=2.0,
            alpha=0.95
        )
    except Exception:
        return


def create_scatter_plot_panel(
    file_df: pd.DataFrame,
    plot_specs: List[Tuple[str, str]],
    output_png: str,
    output_pdf: str,
    figure_title: Optional[str] = None
) -> None:
    x_col = "file_commit_count_intro_to_payment"

    required_cols = [x_col, "group"] + [m[0] for m in plot_specs]
    missing_cols = [c for c in required_cols if c not in file_df.columns]

    if missing_cols:
        print(f"[PLOT-SKIP] Missing columns for scatter plot: {missing_cols}")
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.2),
        constrained_layout=False
    )

    legend_handles = []
    legend_labels = []

    group_order = ["Self-Fixed", "Non-Self-Fixed"]

    for ax, (metric_col, title) in zip(axes, plot_specs):
        ax.axhline(
            0,
            linestyle="--",
            linewidth=1.2,
            color="#1f77b4",
            alpha=0.9
        )

        for group in group_order:
            subset = file_df[file_df["group"] == group].copy()

            if subset.empty:
                continue

            temp = subset[[x_col, metric_col]].copy()
            temp[x_col] = pd.to_numeric(temp[x_col], errors="coerce")
            temp[metric_col] = pd.to_numeric(temp[metric_col], errors="coerce")
            temp = temp.dropna()

            if temp.empty:
                continue

            x = temp[x_col].to_numpy(dtype=float)
            y = temp[metric_col].to_numpy(dtype=float)

            color = GROUP_COLORS.get(group, None)
            label = GROUP_PLOT_LABELS.get(group, group)

            scatter = ax.scatter(
                x,
                y,
                s=PLOT_MARKER_SIZE,
                alpha=PLOT_ALPHA,
                color=color,
                label=label
            )

            add_regression_line(
                ax=ax,
                x=x,
                y=y,
                color=color
            )

            if label not in legend_labels:
                legend_handles.append(scatter)
                legend_labels.append(label)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("File commit count", fontsize=12)
        ax.set_ylabel("Payment - Introduction", fontsize=12)

        if PLOT_SHOW_GRID:
            ax.grid(
                True,
                linestyle="--",
                linewidth=0.8,
                alpha=0.35
            )

    if figure_title:
        fig.suptitle(figure_title, fontsize=15, y=1.03)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.01)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(
        output_png,
        dpi=PLOT_DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved scatter plot PNG to:")
    print(output_png)

    print("\nSaved scatter plot PDF to:")
    print(output_pdf)


def create_dependency_and_complexity_scatter_plots(file_df: pd.DataFrame) -> None:
    dependency_specs = [
        ("delta_fan_in", "Δ Fan-In"),
        ("delta_fan_out", "Δ Fan-Out"),
    ]

    complexity_specs = [
        ("delta_cyclomatic_complexity", "Δ Cyclomatic Complexity"),
        ("delta_cognitive_complexity", "Δ Cognitive Complexity"),
    ]

    create_scatter_plot_panel(
        file_df=file_df,
        plot_specs=dependency_specs,
        output_png=OUTPUT_DEPENDENCY_SCATTER_PNG,
        output_pdf=OUTPUT_DEPENDENCY_SCATTER_PDF,
        figure_title=None
    )

    create_scatter_plot_panel(
        file_df=file_df,
        plot_specs=complexity_specs,
        output_png=OUTPUT_COMPLEXITY_SCATTER_PNG,
        output_pdf=OUTPUT_COMPLEXITY_SCATTER_PDF,
        figure_title=None
    )


# ============================================================
# Sanity Checks
# ============================================================

def print_sanity_checks(
    file_df: pd.DataFrame,
    repo_selection_log_df: pd.DataFrame
) -> None:
    print("\n============================================================")
    print("Sanity Checks")
    print("============================================================")

    print(f"Total file-level rows: {len(file_df)}")
    print(f"Total repository-selection rows: {len(repo_selection_log_df)}")

    if not repo_selection_log_df.empty:
        print("\nRepository selection status:")
        print(
            repo_selection_log_df["repo_selection_status"]
            .value_counts(dropna=False)
            .to_string()
        )

        print("\nSelected repository distribution:")
        print(
            repo_selection_log_df["repo_folder"]
            .value_counts(dropna=False)
            .to_string()
        )

        failed_repo_df = repo_selection_log_df[
            repo_selection_log_df["repo_ready"] != True
        ]

        if not failed_repo_df.empty:
            print("\nRepository selection failures:")
            print(
                failed_repo_df[
                    [
                        "Key",
                        "project_key",
                        "repo_folder",
                        "intro_hash",
                        "payment_hash",
                        "intro_commit_found",
                        "payment_commit_found",
                        "repo_selection_status"
                    ]
                ].head(50).to_string(index=False)
            )

    if file_df.empty:
        print("\nNo file-level rows generated.")
        print("============================================================\n")
        return

    if "source_csv_file" in file_df.columns:
        print("\nRows by source CSV:")
        print(file_df["source_csv_file"].value_counts(dropna=False).to_string())

    if "indicator" in file_df.columns:
        print("\nIndicator distribution:")
        print(file_df["indicator"].value_counts(dropna=False).to_string())

    if "group" in file_df.columns:
        print("\nGroup distribution:")
        print(file_df["group"].value_counts(dropna=False).to_string())

    if "repo_folder" in file_df.columns:
        print("\nRepository distribution in file-level rows:")
        print(file_df["repo_folder"].value_counts(dropna=False).to_string())

    if "tracking_status" in file_df.columns:
        print("\nTracking status distribution:")
        print(file_df["tracking_status"].value_counts(dropna=False).to_string())

    if "deleted" in file_df.columns:
        print("\nDeleted distribution:")
        print(file_df["deleted"].value_counts(dropna=False).to_string())

    if "exists_at_intro" in file_df.columns:
        print("\nExists at intro distribution:")
        print(file_df["exists_at_intro"].value_counts(dropna=False).to_string())

    if "rename_count" in file_df.columns:
        print("\nRename count summary:")
        print(file_df["rename_count"].describe().to_string())

    if "file_commit_count_intro_to_payment" in file_df.columns:
        print("\nFile commit count summary:")
        print(
            file_df["file_commit_count_intro_to_payment"]
            .describe()
            .to_string()
        )

    metric_cols = [
        "delta_fan_in",
        "delta_fan_out",
        "delta_cyclomatic_complexity",
        "delta_cognitive_complexity",
    ]

    existing_metric_cols = [c for c in metric_cols if c in file_df.columns]

    if existing_metric_cols:
        print("\nMissing values in metric deltas:")
        print(file_df[existing_metric_cols].isna().sum().to_string())

    print("============================================================\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("============================================================")
    print("RQ3 File-Level Change Analysis - FAST + Cassandra Fallback + 2 Figures")
    print("============================================================")
    print(f"Input directory       : {INPUT_DIR}")
    print(f"CSV pattern           : {CSV_PATTERN}")
    print(f"Repositories root     : {REPOS_ROOT}")
    print(f"Auto clone repository : {AUTO_CLONE_REPOSITORY}")
    print(f"Output directory      : {OUTPUT_DIR}")
    print(f"Indicator filter      : {FILTER_INDICATOR}")
    print("Allowed repositories  :")
    for repo in sorted(ALLOWED_REPO_FOLDERS):
        print(f"  - {repo}")
    print("============================================================\n")

    file_df, repo_selection_log_df = build_file_level_rq3_dataset(
        INPUT_DIR,
        CSV_PATTERN
    )

    repo_selection_log_df.to_csv(OUTPUT_REPO_SELECTION_LOG_CSV, index=False)

    if file_df.empty:
        print_sanity_checks(file_df, repo_selection_log_df)

        print("\nNo file-level rows generated.")
        print("Please check:")
        print("1. Whether indicator='viomod' exists in the input dataset.")
        print("2. Whether Intro_Affected_Files contains Java files.")
        print("3. Whether each Jira key maps to the expected target repository.")
        print("4. Whether Intro Hash and Payment Hash exist in the mapped repository or Cassandra fallback.")
        print("5. Check repository selection log:")
        print(OUTPUT_REPO_SELECTION_LOG_CSV)

        raise SystemExit(0)

    print_sanity_checks(file_df, repo_selection_log_df)

    file_df.to_csv(OUTPUT_FILE_LEVEL_CSV, index=False)

    print("Saved file-level dataset to:")
    print(OUTPUT_FILE_LEVEL_CSV)

    print("\nSaved repository selection log to:")
    print(OUTPUT_REPO_SELECTION_LOG_CSV)

    spearman_df = spearman_analysis(file_df)
    spearman_df.to_csv(OUTPUT_SPEARMAN_CSV, index=False)

    print("\nSaved Spearman correlation results with effect size to:")
    print(OUTPUT_SPEARMAN_CSV)

    summary_df = descriptive_summary(file_df)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print("\nSaved descriptive summary to:")
    print(OUTPUT_SUMMARY_CSV)

    create_dependency_and_complexity_scatter_plots(file_df)

    print("\n============================================================")
    print("Preview: File-Level Dataset")
    print("============================================================")
    print(file_df.head(10).to_string(index=False))

    print("\n============================================================")
    print("Preview: Repository Selection Log")
    print("============================================================")
    print(repo_selection_log_df.head(10).to_string(index=False))

    print("\n============================================================")
    print("Preview: Spearman Results with Effect Size")
    print("============================================================")
    print(spearman_df.to_string(index=False))

    print("\nDone.")