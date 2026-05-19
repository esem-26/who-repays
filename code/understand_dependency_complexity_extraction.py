import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Optional, Any

import lizard
import pandas as pd
from tqdm import tqdm


# ============================================================
# SciTools Understand configuration
# ============================================================

UNDERSTAND_BIN_DIR = os.environ.get("SCITOOLS_UND_BIN", "/opt/scitools/bin/linux64")
UNDERSTAND_PY_DIR = os.path.join(UNDERSTAND_BIN_DIR, "Python")

UNDERSTAND_PY_CANDIDATES = [
    UNDERSTAND_PY_DIR,
    "/home/edsu/scitools/bin/linux64/Python",
    "/home/edsu/scitools/Python",
    "/opt/scitools/python",
    "/opt/scitools/understand/python",
]


# ============================================================
# Input / output
# ============================================================

DATA_DIRECTORY = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/"

TRACED_DATASET_INPUT_PATH = os.path.join(
    DATA_DIRECTORY,
    "ATD-DATA/SELF-FIXED/csv_by_project/CAMEL.csv",
)

METRICS_DATASET_OUTPUT_PATH = os.path.join(
    DATA_DIRECTORY,
    "ATD-DATA/SELF-FIXED/METRICS/CAMEL-UNDERSTAND-METRICS-CC-COMPLEXITY-FINAL.csv",
)


# ============================================================
# Repository root directory
# ============================================================

REPOS_ROOT_DIRECTORY = os.path.join(
    DATA_DIRECTORY,
    "ATD-DATA/repos",
)

PROJECT_REPO_FOLDER_MAPPING = {
    "SPARK": "spark",
    "CAMEL": "camel",
    "CASSANDRA": "cassandra",
    "KAFKA": "kafka",
    "AMQ": "activemq",
    "DRILL": "drill",
    "SOLR": "solr",
    "LUCENE": "lucene",
    "GEODE": "geode",
    "NETBEANS": "netbeans",
}


# ============================================================
# Custom temp directory
# ============================================================

CUSTOM_TMPDIR = os.environ.get(
    "WORK_TMPDIR",
    "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/tmp",
)

os.makedirs(CUSTOM_TMPDIR, exist_ok=True)
tempfile.tempdir = CUSTOM_TMPDIR


# ============================================================
# Project configuration
# ============================================================

PROJECT_LANGUAGE_MAPPING = {
    "SPARK": "scala",
    "CAMEL": "java",
    "CASSANDRA": "java",
    "KAFKA": "java",
    "AMQ": "java",
    "DRILL": "java",
    "SOLR": "java",
    "LUCENE": "java",
    "GEODE": "java",
    "NETBEANS": "java",
}

PROJECT_DEFAULT_BRANCH_MAPPING = {
    "SPARK": "master",
    "CAMEL": "main",
    "CASSANDRA": "trunk",
    "KAFKA": "trunk",
    "AMQ": "main",
    "DRILL": "master",
    "SOLR": "main",
    "LUCENE": "main",
    "GEODE": "develop",
    "NETBEANS": "master",
}


# ============================================================
# Globals
# ============================================================

understand: Optional[Any] = None
GIT_EXECUTABLE_PATH = None


understand_metric_suffixes = [
    "ClassDetails_JSON",
    "FileDetails_JSON",
    "Status",
]

git_diff_metric_column_names = [
    "Payment_TotalFilesChangedInScope",
    "Payment_TotalLinesAddedInScope",
    "Payment_TotalLinesDeletedInScope",
    "Payment_FileChangesDetails_JSON",
    "Payment_FileChangesStatus",
]

new_commit_level_metric_suffixes = [
    "FilesDeletedByCommit",
    "TotalFilesInRepo",
]

custom_added_columns = [
    "Payment_MissingIntroFiles",
    "Payment_Path_Map_JSON",
]


# ============================================================
# Utility functions
# ============================================================

def get_repository_path_from_project_key(project_key: str) -> str:
    """
    Build repository path from project key using REPOS_ROOT_DIRECTORY.
    """
    project_key = str(project_key).upper().strip()

    repo_folder_name = PROJECT_REPO_FOLDER_MAPPING.get(project_key)

    if not repo_folder_name:
        raise ValueError(
            f"No repository folder mapping found for project key: {project_key}"
        )

    repository_path = os.path.join(REPOS_ROOT_DIRECTORY, repo_folder_name)
    repository_path = os.path.normpath(repository_path)

    return repository_path


def _normalize_path_for_keys(path_str):
    """
    Normalize path for dictionary lookup.
    """
    if not path_str or pd.isna(path_str):
        return ""

    return (
        os.path.normpath(str(path_str).replace("\\", "/"))
        .replace(os.sep, "/")
        .lower()
    )


def _split_semicolon_paths(paths_string) -> list[str]:
    """
    Split semicolon-separated file paths.
    """
    if pd.isna(paths_string) or not str(paths_string).strip():
        return []

    return [
        p.strip()
        for p in str(paths_string).split(";")
        if p and p.strip()
    ]


def _filter_java_files(file_paths: list[str]) -> list[str]:
    """
    Keep only Java files.
    """
    return [
        p for p in file_paths
        if p.lower().endswith(".java")
    ]


def _execute_command(command_args, **kwargs):
    """
    Robust subprocess runner.

    Important:
    - This wrapper prints the command.
    - It preserves compatibility with old calls that pass check=False.
    - It does not force check=True.
    """
    print(f"Running: {' '.join(command_args)} ...")

    kwargs.setdefault("stdout", subprocess.PIPE)
    kwargs.setdefault("stderr", subprocess.PIPE)
    kwargs.setdefault("text", True)
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("errors", "ignore")

    return subprocess.run(command_args, **kwargs)


def setup_git_executable():
    """
    Locate git executable.
    """
    global GIT_EXECUTABLE_PATH

    GIT_EXECUTABLE_PATH = shutil.which("git")

    if GIT_EXECUTABLE_PATH is None:
        print("Git executable not found...")
        sys.exit(1)


def setup_understand_environment() -> bool:
    """
    Load SciTools Understand Python API.
    """
    global understand

    os.environ.setdefault("SCITOOLS_UND_BIN", UNDERSTAND_BIN_DIR)

    if UNDERSTAND_BIN_DIR not in os.environ.get("PATH", ""):
        os.environ["PATH"] = UNDERSTAND_BIN_DIR + os.pathsep + os.environ.get("PATH", "")

    try:
        import understand as und_module  # type: ignore

        understand = und_module
        print(f"✅ Understand Python API loaded from {und_module.__file__}")
        return True

    except ImportError:
        pass

    for candidate in UNDERSTAND_PY_CANDIDATES:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)

            try:
                import understand as und_module  # type: ignore

                understand = und_module
                print(f"✅ Understand Python API loaded from {und_module.__file__}")
                return True

            except ImportError:
                sys.path.pop(0)

    print("❌ Understand Python API not found. Please check PYTHONPATH or installation path.")
    print("   Tried paths:\n   " + "\n   ".join(UNDERSTAND_PY_CANDIDATES))

    understand = None
    return False


def get_und_executable_path() -> str:
    """
    Return path to und executable.
    """
    und_executable = shutil.which("und")

    if und_executable:
        return und_executable

    fallback = os.path.join(UNDERSTAND_BIN_DIR, "und")

    if os.path.exists(fallback):
        return fallback

    return "und"


def run_git_command(
    repository_path,
    arguments_list,
    check_for_errors=True,
    check=None,
    verbose=True,
):
    """
    Run Git command in repository_path.

    This function is intentionally compatible with both:
    - run_git_command(..., check_for_errors=False)
    - run_git_command(..., check=False)

    Returns
    -------
    str
        stdout stripped.
    """
    if check is not None:
        check_for_errors = check

    if GIT_EXECUTABLE_PATH is None:
        raise RuntimeError("GIT_EXECUTABLE_PATH is not set. Call setup_git_executable() first.")

    command = [GIT_EXECUTABLE_PATH, "-C", repository_path] + arguments_list

    if verbose:
        print(f"Running: {' '.join(command)} ...")

    command_result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )

    if check_for_errors and command_result.returncode != 0:
        print("\nGit command failed")
        print(f"Repository : {repository_path}")
        print(f"Command    : {' '.join(command)}")
        print(f"Return code: {command_result.returncode}")
        print("STDOUT:")
        print(command_result.stdout if command_result.stdout else "<empty>")
        print("STDERR:")
        print(command_result.stderr if command_result.stderr else "<empty>")

        raise subprocess.CalledProcessError(
            command_result.returncode,
            command,
            output=command_result.stdout,
            stderr=command_result.stderr,
        )

    return command_result.stdout.strip()


def run_git_command_result(
    repository_path,
    arguments_list,
    check_for_errors=True,
    verbose=True,
):
    """
    Run Git command and return CompletedProcess instead of stdout.
    Useful when return code is needed.
    """
    if GIT_EXECUTABLE_PATH is None:
        raise RuntimeError("GIT_EXECUTABLE_PATH is not set. Call setup_git_executable() first.")

    command = [GIT_EXECUTABLE_PATH, "-C", repository_path] + arguments_list

    if verbose:
        print(f"Running: {' '.join(command)} ...")

    command_result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )

    if check_for_errors and command_result.returncode != 0:
        print("\nGit command failed")
        print(f"Repository : {repository_path}")
        print(f"Command    : {' '.join(command)}")
        print(f"Return code: {command_result.returncode}")
        print("STDOUT:")
        print(command_result.stdout if command_result.stdout else "<empty>")
        print("STDERR:")
        print(command_result.stderr if command_result.stderr else "<empty>")

        raise subprocess.CalledProcessError(
            command_result.returncode,
            command,
            output=command_result.stdout,
            stderr=command_result.stderr,
        )

    return command_result


def safe_git_clean(repository_path: str, project_key: str = "", context: str = "") -> bool:
    """
    Clean untracked files safely.

    First tries:
      git clean -fdx

    If it fails, retries:
      git clean -ffdx

    Returns True if clean succeeds, otherwise False.
    """
    label = f"{project_key} {context}".strip()

    try:
        run_git_command(repository_path, ["clean", "-fdx"])
        return True

    except subprocess.CalledProcessError as e:
        print(f"Warning: git clean -fdx failed for {label}.")
        print("Retrying with git clean -ffdx ...")
        print(f"stdout: {getattr(e, 'stdout', '')}")
        print(f"stderr: {getattr(e, 'stderr', '')}")

        try:
            run_git_command(repository_path, ["clean", "-ffdx"])
            return True

        except subprocess.CalledProcessError as e2:
            print(f"Warning: git clean -ffdx also failed for {label}.")
            print(f"stdout: {getattr(e2, 'stdout', '')}")
            print(f"stderr: {getattr(e2, 'stderr', '')}")
            print(
                "Continuing because git reset --hard may already have succeeded. "
                "Some untracked files may remain in the repository."
            )
            return False


def abort_possible_git_operation_states(repository_path: str) -> None:
    """
    Abort incomplete rebase/merge/cherry-pick states if present.
    Safe to call even when none are active.
    """
    for args in [
        ["rebase", "--abort"],
        ["merge", "--abort"],
        ["cherry-pick", "--abort"],
    ]:
        try:
            run_git_command(repository_path, args, check_for_errors=False, verbose=False)
        except Exception:
            pass


def _is_valid_commit_hash(commit_hash: str) -> bool:
    """
    Basic validation for commit hash string.
    """
    if not commit_hash:
        return False

    commit_hash = str(commit_hash).strip()

    if len(commit_hash) < 7:
        return False

    if commit_hash.lower() in ["nan", "<na>", "none"]:
        return False

    if "unknown" in commit_hash.lower():
        return False

    return True


def _is_json_content_present(value_to_check):
    """
    Check whether a JSON-like metric column already contains content.
    """
    if pd.isna(value_to_check):
        return False

    stripped_value = str(value_to_check).strip()

    return bool(stripped_value and stripped_value not in ["[]", "nan", "NaN", "<NA>", ""])


def ensure_enough_disk_space(path: str, min_free_gb: float = 10.0) -> None:
    """
    Ensure enough free disk space before running Understand.
    """
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)

    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Not enough disk space at {path}. "
            f"Free: {free_gb:.2f} GB, required at least: {min_free_gb:.2f} GB."
        )


def _clear_scitools_db_cache_on_linux(max_gb: float | None = None) -> None:
    """
    Clear SciTools Understand DB cache.
    Checks both current user cache and root cache.
    """
    if os.name == "nt":
        return

    candidate_caches = [
        os.path.expanduser("~/.config/SciTools/Db/"),
        "/root/.config/SciTools/Db/",
    ]

    def _dir_size_bytes(p: str) -> int:
        total = 0

        for root, _, files in os.walk(p, onerror=lambda e: None):
            for f in files:
                fp = os.path.join(root, f)

                try:
                    total += os.path.getsize(fp)
                except Exception:
                    pass

        return total

    for cache in candidate_caches:
        if not os.path.isdir(cache):
            continue

        if max_gb is not None:
            try:
                size_gb = _dir_size_bytes(cache) / (1024 ** 3)

                if size_gb <= max_gb:
                    continue

            except Exception:
                pass

        print(f"Clearing SciTools cache: {cache}")

        for ent in os.listdir(cache):
            try:
                shutil.rmtree(os.path.join(cache, ent), ignore_errors=True)
            except Exception:
                pass


def checkout_commit_safely(repository_path: str, commit_hash: str, context: str = "") -> str:
    """
    Checkout a commit and verify that HEAD is really at the expected commit.
    """
    if not _is_valid_commit_hash(commit_hash):
        raise ValueError(f"Invalid commit hash for {context}: {commit_hash}")

    abort_possible_git_operation_states(repository_path)

    run_git_command(repository_path, ["reset", "--hard"])
    safe_git_clean(repository_path, context=context)

    run_git_command(repository_path, ["checkout", "--detach", commit_hash])

    run_git_command(repository_path, ["reset", "--hard"])
    safe_git_clean(repository_path, context=context)

    actual_head = run_git_command(repository_path, ["rev-parse", "HEAD"])
    expected_full_hash = run_git_command(repository_path, ["rev-parse", commit_hash])

    if actual_head != expected_full_hash:
        raise RuntimeError(
            f"Checkout verification failed for {context}. "
            f"Expected {expected_full_hash}, but HEAD is {actual_head}"
        )

    print(f"{context} checked out correctly at {actual_head}")

    return actual_head


def verify_files_exist_in_worktree(
    repository_path: str,
    file_paths: list[str],
    context: str = "",
) -> list[str]:
    """
    Verify which files physically exist in the current working tree.
    """
    existing_files = []
    missing_files = []

    for rel_path in file_paths:
        abs_path = os.path.join(repository_path, rel_path)

        if os.path.exists(abs_path):
            existing_files.append(rel_path)
        else:
            missing_files.append(rel_path)

    if existing_files:
        print(f"{context} existing files in working tree:")
        for p in existing_files:
            print(f"  FOUND: {p}")

    if missing_files:
        print(f"{context} missing files in working tree:")
        for p in missing_files:
            print(f"  MISSING: {p}")

    return existing_files


def print_debug_head_and_files(
    repository_path: str,
    file_paths: list[str],
    context: str = "",
) -> None:
    """
    Print current HEAD and file existence for debugging.
    """
    try:
        head = run_git_command(repository_path, ["rev-parse", "HEAD"])
        print(f"{context} DEBUG HEAD: {head}")
    except Exception as e:
        print(f"{context} DEBUG HEAD failed: {e}")

    for p in file_paths:
        print(
            f"{context} DEBUG FILE EXISTS: {p} -> "
            f"{os.path.exists(os.path.join(repository_path, p))}"
        )


# ============================================================
# Cognitive Complexity Approximation for Java
# ============================================================

def _strip_java_comments_preserving_lines(lines: list[str]) -> list[str]:
    """
    Remove Java block comments and line comments while preserving line count.
    This is a lightweight cleaner, not a full Java lexer.
    """
    cleaned_lines = []
    in_block_comment = False

    for line in lines:
        i = 0
        cleaned = ""

        while i < len(line):
            if in_block_comment:
                end_idx = line.find("*/", i)

                if end_idx == -1:
                    i = len(line)
                else:
                    in_block_comment = False
                    i = end_idx + 2

            else:
                line_comment_idx = line.find("//", i)
                block_comment_idx = line.find("/*", i)

                if line_comment_idx != -1 and (
                    block_comment_idx == -1 or line_comment_idx < block_comment_idx
                ):
                    cleaned += line[i:line_comment_idx]
                    break

                if block_comment_idx != -1:
                    cleaned += line[i:block_comment_idx]
                    in_block_comment = True
                    i = block_comment_idx + 2
                else:
                    cleaned += line[i:]
                    break

        cleaned_lines.append(cleaned)

    return cleaned_lines


def _remove_java_string_literals(line: str) -> str:
    """
    Remove string and character literals to avoid false keyword matches.
    """
    line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
    line = re.sub(r"'(?:\\.|[^'\\])'", "''", line)
    return line


def _count_logical_operator_complexity(line: str) -> int:
    """
    Approximate cognitive complexity contribution from logical operator sequences.
    """
    operators = re.findall(r"&&|\|\|", line)

    if not operators:
        return 0

    groups = 1

    for i in range(1, len(operators)):
        if operators[i] != operators[i - 1]:
            groups += 1

    return groups


def _is_method_declaration_line(line: str, method_name: str | None) -> bool:
    """
    Avoid counting the method declaration as recursion.
    """
    if not method_name:
        return False

    declaration_keywords = (
        "public",
        "protected",
        "private",
        "static",
        "final",
        "synchronized",
        "native",
        "abstract",
        "strictfp",
    )

    return bool(
        re.search(rf"\b{re.escape(method_name)}\s*\(", line)
        and any(re.search(rf"\b{k}\b", line) for k in declaration_keywords)
    )


def calculate_java_cognitive_complexity_approx(
    file_lines: list[str],
    start_line: int | None,
    end_line: int | None,
    method_name: str | None = None,
) -> int:
    """
    Approximate Cognitive Complexity for a Java method/function.
    This is not the official SonarQube implementation.
    """
    if start_line is None or end_line is None:
        return 0

    if start_line <= 0 or end_line <= 0 or end_line < start_line:
        return 0

    method_lines = file_lines[max(start_line - 1, 0): end_line]
    method_lines = _strip_java_comments_preserving_lines(method_lines)

    cognitive_complexity = 0
    nesting = 0
    brace_depth = 0
    nesting_stack: list[int] = []

    for raw_line in method_lines:
        line = _remove_java_string_literals(raw_line).strip()

        if not line:
            continue

        while nesting_stack and brace_depth < nesting_stack[-1]:
            nesting_stack.pop()
            nesting = max(0, nesting - 1)

        if line.startswith("@") or line.startswith("import ") or line.startswith("package "):
            brace_depth += line.count("{") - line.count("}")
            continue

        starts_nested_structure = False

        if re.search(r"\belse\s+if\s*\(", line):
            cognitive_complexity += 1
            cognitive_complexity += _count_logical_operator_complexity(line)

        else:
            if re.search(r"\bif\s*\(", line):
                cognitive_complexity += 1 + nesting
                cognitive_complexity += _count_logical_operator_complexity(line)
                starts_nested_structure = True

            if re.search(r"\b(for|while)\s*\(", line):
                cognitive_complexity += 1 + nesting
                cognitive_complexity += _count_logical_operator_complexity(line)
                starts_nested_structure = True

            if re.search(r"\bdo\b", line):
                cognitive_complexity += 1 + nesting
                starts_nested_structure = True

            if re.search(r"\bswitch\s*\(", line):
                cognitive_complexity += 1 + nesting
                starts_nested_structure = True

            if re.search(r"\bcatch\s*\(", line):
                cognitive_complexity += 1 + nesting
                cognitive_complexity += _count_logical_operator_complexity(line)
                starts_nested_structure = True

            if "?" in line and ":" in line:
                cognitive_complexity += 1 + nesting
                cognitive_complexity += _count_logical_operator_complexity(line)

        if re.search(r"\b(break|continue)\s*;", line):
            cognitive_complexity += 1

        if method_name and re.search(rf"\b{re.escape(method_name)}\s*\(", line):
            if not _is_method_declaration_line(line, method_name):
                cognitive_complexity += 1

        opening_braces = line.count("{")
        closing_braces = line.count("}")

        if starts_nested_structure and opening_braces > closing_braces:
            nesting_stack.append(brace_depth + opening_braces - closing_braces)
            nesting += 1

        brace_depth += opening_braces - closing_braces

    return cognitive_complexity


# ============================================================
# Git path / rename mapping
# ============================================================

def git_file_exists_at_commit(repository_path: str, commit_hash: str, rel_path: str) -> bool:
    """
    Return True if rel_path exists in commit_hash tree.
    """
    try:
        run_git_command(
            repository_path,
            ["cat-file", "-e", f"{commit_hash}:{rel_path}"],
        )
        return True

    except Exception:
        return False


def follow_rename_to_commit(
    repository_path: str,
    old_rel_path: str,
    start_commit: str,
    end_commit: str,
) -> Optional[str]:
    """
    Map intro file path to its equivalent path at payment commit.
    """
    if not old_rel_path:
        return None

    if git_file_exists_at_commit(repository_path, end_commit, old_rel_path):
        return old_rel_path

    try:
        diff_output = run_git_command(
            repository_path,
            [
                "diff",
                "--name-status",
                "--find-renames=20%",
                start_commit,
                end_commit,
                "--",
                old_rel_path,
            ],
            check_for_errors=False,
        )

        for line in diff_output.splitlines():
            parts = line.split("\t")

            if not parts:
                continue

            status = parts[0]

            if status.startswith("R") and len(parts) >= 3:
                old_path = parts[1]
                new_path = parts[2]

                if _normalize_path_for_keys(old_path) == _normalize_path_for_keys(old_rel_path):
                    if git_file_exists_at_commit(repository_path, end_commit, new_path):
                        return new_path

            if status == "D" and len(parts) >= 2:
                deleted_path = parts[1]

                if _normalize_path_for_keys(deleted_path) == _normalize_path_for_keys(old_rel_path):
                    return None

    except Exception:
        pass

    try:
        cmd = [
            GIT_EXECUTABLE_PATH,
            "-C",
            repository_path,
            "log",
            "--reverse",
            "--find-renames=20%",
            "--name-status",
            "--format=%x00%H",
            f"{start_commit}..{end_commit}",
            "--",
            old_rel_path,
        ]

        proc = _execute_command(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )

        out = proc.stdout

        if not out:
            return None

        current_path = old_rel_path
        chunks = out.split("\x00")

        for chunk in chunks[1:]:
            lines = [ln for ln in chunk.splitlines() if ln.strip()]

            if not lines:
                continue

            if len(lines[0]) in (7, 40) and all(
                c in "0123456789abcdef"
                for c in lines[0].lower()
            ):
                lines = lines[1:]

            for ln in lines:
                parts = ln.split("\t")

                if not parts:
                    continue

                status = parts[0]

                if status.startswith("R") and len(parts) >= 3:
                    old_path = parts[1]
                    new_path = parts[2]

                    if _normalize_path_for_keys(old_path) == _normalize_path_for_keys(current_path):
                        current_path = new_path

                elif status == "D" and len(parts) >= 2:
                    deleted_path = parts[1]

                    if _normalize_path_for_keys(deleted_path) == _normalize_path_for_keys(current_path):
                        return None

        if git_file_exists_at_commit(repository_path, end_commit, current_path):
            return current_path

    except Exception:
        pass

    return None


def map_intro_paths_to_payment_paths(
    repository_path: str,
    introduction_files_list: list[str],
    introduction_commit_hash: str,
    payment_commit_hash: str,
) -> tuple[dict[str, Optional[str]], list[str], list[str]]:
    """
    Map intro file paths to valid paths at payment commit.
    """
    path_map: dict[str, Optional[str]] = {}
    payment_paths: list[str] = []
    missing_intro_files: list[str] = []

    for intro_path in introduction_files_list:
        mapped_path = follow_rename_to_commit(
            repository_path=repository_path,
            old_rel_path=intro_path,
            start_commit=introduction_commit_hash,
            end_commit=payment_commit_hash,
        )

        path_map[intro_path] = mapped_path

        if mapped_path:
            payment_paths.append(mapped_path)
        else:
            missing_intro_files.append(intro_path)

    payment_paths = sorted(set(payment_paths))

    return path_map, payment_paths, missing_intro_files


# ============================================================
# Repository restore
# ============================================================

def restore_git_state(repository_path, project_key, context_message=""):
    """
    Restore repository to default branch.

    This version is robust for Kafka and other large Apache repositories.
    It does not fail only because git clean cannot remove untracked files.
    """
    project_key = str(project_key).upper().strip()
    default_branch = PROJECT_DEFAULT_BRANCH_MAPPING.get(project_key)

    if not default_branch:
        raise RuntimeError(f"No default branch configured for {project_key}")

    if not os.path.isdir(repository_path):
        raise FileNotFoundError(
            f"Repository path does not exist for {project_key}: {repository_path}"
        )

    git_dir = os.path.join(repository_path, ".git")
    if not os.path.exists(git_dir):
        raise RuntimeError(
            f"Repository path is not a valid Git repository for {project_key}:\n"
            f"{repository_path}\n"
            f"Missing .git directory."
        )

    print(f"{context_message} Resetting repository to {default_branch}...")

    optional_git_configs = [
        ["config", "core.feature.manyFiles", "true"],
        ["config", "core.longpaths", "true"],
        ["config", "core.preloadIndex", "true"],
        ["config", "gc.auto", "0"],
    ]

    for config_args in optional_git_configs:
        try:
            run_git_command(repository_path, config_args)
        except subprocess.CalledProcessError as e:
            print(
                f"Warning: optional Git config failed for {project_key}: "
                f"git {' '.join(config_args)}"
            )
            print(f"stderr: {getattr(e, 'stderr', '')}")

    abort_possible_git_operation_states(repository_path)

    run_git_command(repository_path, ["reset", "--hard"])
    safe_git_clean(repository_path, project_key=project_key, context="before checkout default branch")

    checkout_success = False

    try:
        run_git_command(repository_path, ["checkout", default_branch])
        checkout_success = True

    except subprocess.CalledProcessError:
        print(
            f"Warning: local branch {default_branch} checkout failed for {project_key}. "
            f"Trying origin/{default_branch}..."
        )

        try:
            run_git_command(
                repository_path,
                ["checkout", "-B", default_branch, f"origin/{default_branch}"],
            )
            checkout_success = True

        except subprocess.CalledProcessError as e:
            print(f"Failed to checkout default branch for {project_key}: {default_branch}")
            print(f"stderr: {getattr(e, 'stderr', '')}")

    if not checkout_success:
        fallback_branches = ["trunk", "main", "master", "develop"]

        for branch_name in fallback_branches:
            if branch_name == default_branch:
                continue

            try:
                run_git_command(repository_path, ["checkout", branch_name])
                default_branch = branch_name
                checkout_success = True
                break

            except subprocess.CalledProcessError:
                try:
                    run_git_command(
                        repository_path,
                        ["checkout", "-B", branch_name, f"origin/{branch_name}"],
                    )
                    default_branch = branch_name
                    checkout_success = True
                    break

                except subprocess.CalledProcessError:
                    continue

    if not checkout_success:
        raise RuntimeError(
            f"Could not checkout any default branch for {project_key}. "
            f"Tried configured branch and fallbacks."
        )

    run_git_command(repository_path, ["reset", "--hard"])
    safe_git_clean(repository_path, project_key=project_key, context="after checkout default branch")

    upstream_result = run_git_command_result(
        repository_path,
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        check_for_errors=False,
        verbose=False,
    )

    if upstream_result.returncode == 0 and upstream_result.stdout.strip():
        try:
            run_git_command(repository_path, ["pull", "--ff-only"])
        except subprocess.CalledProcessError as e:
            print(
                f"Warning: git pull --ff-only failed for {project_key}. "
                f"Continuing with local branch {default_branch}."
            )
            print(f"stderr: {getattr(e, 'stderr', '')}")
    else:
        print(
            f"Warning: branch {default_branch} for {project_key} has no upstream. "
            f"Skipping git pull."
        )

    print(f"{project_key}: restored to branch {default_branch}")


# ============================================================
# Understand analysis helpers
# ============================================================

def _ensure_understand_project_analyzed(source_root_directory, understand_db_path, project_key):
    """
    Create, add, and analyze Understand database.
    """
    language = PROJECT_LANGUAGE_MAPPING.get(str(project_key).upper(), "guess")
    und_executable = get_und_executable_path()

    db_parent_dir = os.path.dirname(understand_db_path)
    os.makedirs(db_parent_dir, exist_ok=True)

    if os.path.exists(understand_db_path):
        if os.path.isdir(understand_db_path):
            shutil.rmtree(understand_db_path, ignore_errors=True)
        else:
            os.remove(understand_db_path)

    print(f"Creating UDB at: {understand_db_path}")
    print(f"Source root: {source_root_directory}")
    print(f"Language: {language}")

    ensure_enough_disk_space(db_parent_dir, min_free_gb=10.0)

    create_result = _execute_command(
        [und_executable, "create", "-languages", language, understand_db_path],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    if create_result.returncode != 0:
        print("Und create failed.")
        print("STDOUT:")
        print(create_result.stdout)
        print("STDERR:")
        print(create_result.stderr)

        raise subprocess.CalledProcessError(
            create_result.returncode,
            create_result.args,
            output=create_result.stdout,
            stderr=create_result.stderr,
        )

    add_result = _execute_command(
        [und_executable, "add", source_root_directory, understand_db_path],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    if add_result.returncode not in (0, 1):
        print("Und add failed.")
        print("STDOUT:")
        print(add_result.stdout)
        print("STDERR:")
        print(add_result.stderr)

        raise subprocess.CalledProcessError(
            add_result.returncode,
            add_result.args,
            output=add_result.stdout,
            stderr=add_result.stderr,
        )

    ensure_enough_disk_space(db_parent_dir, min_free_gb=10.0)

    analyze_result = _execute_command(
        [und_executable, "analyze", understand_db_path],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    if analyze_result.returncode != 0:
        print("Und analyze failed.")
        print("STDOUT:")
        print(analyze_result.stdout)
        print("STDERR:")
        print(analyze_result.stderr)

        try:
            usage = shutil.disk_usage(db_parent_dir)
            print(f"Disk usage for {db_parent_dir}:")
            print(f"  total: {usage.total / (1024 ** 3):.2f} GB")
            print(f"  used : {usage.used / (1024 ** 3):.2f} GB")
            print(f"  free : {usage.free / (1024 ** 3):.2f} GB")
        except Exception as e:
            print(f"Could not check disk usage: {e}")

        raise subprocess.CalledProcessError(
            analyze_result.returncode,
            analyze_result.args,
            output=analyze_result.stdout,
            stderr=analyze_result.stderr,
        )

    print(f"Analyzed UDB {os.path.basename(understand_db_path)}...")
    return True


def _find_understand_file_entity(understand_database, relative_path: str):
    """
    Find Understand file entity using exact match first, then suffix match.
    """
    normalized_relative_path = _normalize_path_for_keys(relative_path)

    file_entities = list(understand_database.ents("File"))

    exact_map = {
        _normalize_path_for_keys(f.relname()): f
        for f in file_entities
    }

    file_entity = exact_map.get(normalized_relative_path)

    if file_entity:
        return file_entity

    for candidate_file in file_entities:
        candidate_key = _normalize_path_for_keys(candidate_file.relname())

        if candidate_key.endswith(normalized_relative_path):
            return candidate_file

    return None


def _get_define_file_relname(entity):
    """
    Return normalized relative file path where an Understand entity is defined.
    """
    try:
        define_ref = entity.ref("Definein")

        if define_ref and define_ref.file():
            return _normalize_path_for_keys(define_ref.file().relname())

    except Exception:
        pass

    try:
        define_ref = entity.ref("Define")

        if define_ref and define_ref.file():
            return _normalize_path_for_keys(define_ref.file().relname())

    except Exception:
        pass

    return ""


def _get_classes_defined_in_file(understand_database, file_entity):
    """
    More robust way to get Java classes, interfaces, and enums defined in a file.
    """
    target_file_key = _normalize_path_for_keys(file_entity.relname())
    classes = []
    seen_ids = set()

    candidate_kind_queries = [
        "Java Class",
        "Class",
        "Java Interface",
        "Interface",
        "Java Enum",
        "Enum",
        "Java Type",
        "Type",
    ]

    for kind_query in candidate_kind_queries:
        try:
            for ent in understand_database.ents(kind_query):
                try:
                    ent_id = ent.id()
                except Exception:
                    ent_id = ent.longname()

                if ent_id in seen_ids:
                    continue

                define_file_key = _get_define_file_relname(ent)

                if define_file_key == target_file_key:
                    classes.append(ent)
                    seen_ids.add(ent_id)

        except Exception:
            continue

    if not classes:
        for relation in ["Define", "Declare"]:
            try:
                for ent in file_entity.ents(relation):
                    kind = str(ent.kind()).lower()
                    if any(k in kind for k in ["class", "interface", "enum"]):
                        try:
                            ent_id = ent.id()
                        except Exception:
                            ent_id = ent.longname()

                        if ent_id not in seen_ids:
                            classes.append(ent)
                            seen_ids.add(ent_id)
            except Exception:
                pass

    return classes


def _get_methods_defined_in_class(class_entity):
    """
    Return methods/functions/constructors defined in a class entity.
    """
    methods = []
    seen_ids = set()

    method_kind_filters = [
        "Method",
        "Function",
        "Constructor",
        "Destructor",
        "Operator",
        "Property",
        "Java Method",
        "Java Constructor",
    ]

    try:
        for ent in class_entity.ents("Define"):
            kind = str(ent.kind()).lower()

            if any(m.lower() in kind for m in method_kind_filters):
                try:
                    ent_id = ent.id()
                except Exception:
                    ent_id = ent.longname()

                if ent_id not in seen_ids:
                    methods.append(ent)
                    seen_ids.add(ent_id)

    except Exception:
        pass

    try:
        for ent in class_entity.ents(
            "Define",
            "Method, Function, Constructor, Destructor, Operator, Property",
        ):
            try:
                ent_id = ent.id()
            except Exception:
                ent_id = ent.longname()

            if ent_id not in seen_ids:
                methods.append(ent)
                seen_ids.add(ent_id)

    except Exception:
        pass

    return methods


def _get_entity_start_line(entity):
    """
    Get start line for an Understand entity.
    """
    try:
        line = entity.metric("MinLineCode")
        if line:
            return int(line)
    except Exception:
        pass

    for ref_kind in ["Definein", "Define", "Declarein", "Declare"]:
        try:
            ref = entity.ref(ref_kind)
            if ref and ref.line():
                return int(ref.line())
        except Exception:
            pass

    return None


def get_lizard_metrics_with_api(root_directory, file_paths):
    """
    Calculate Lizard cyclomatic complexity and approximate cognitive complexity
    for selected Java files.
    """
    lizard_metrics_output = {}

    for relative_path in file_paths:
        if not relative_path.lower().endswith(".java"):
            continue

        absolute_path = os.path.join(root_directory, relative_path)

        if not os.path.exists(absolute_path):
            print(f"Lizard skipped missing file: {relative_path}")
            continue

        try:
            with open(absolute_path, "r", encoding="utf-8", errors="ignore") as f:
                file_lines = f.readlines()
        except Exception:
            file_lines = []

        try:
            analysis_result = lizard.analyze_file(absolute_path)
        except Exception as e:
            print(f"Lizard failed for {absolute_path}: {e}")
            continue

        key = _normalize_path_for_keys(relative_path)
        function_metrics = []

        for f in analysis_result.function_list:
            start_line = getattr(f, "start_line", None)
            end_line = getattr(f, "end_line", None)

            if end_line is None:
                end_line = start_line

            method_name = f.name.split("::")[-1] if f.name else None

            cognitive_complexity = calculate_java_cognitive_complexity_approx(
                file_lines=file_lines,
                start_line=start_line,
                end_line=end_line,
                method_name=method_name,
            )

            function_metrics.append(
                {
                    "name": f.name,
                    "simple_name": method_name,
                    "ccn": f.cyclomatic_complexity,
                    "cognitive_complexity": cognitive_complexity,
                    "start_line": start_line,
                    "end_line": end_line,
                }
            )

        lizard_metrics_output[key] = function_metrics
        print(f"Lizard extracted {len(function_metrics)} functions from {relative_path}")

    return lizard_metrics_output


def _get_lizard_functions_for_path(
    lizard_metrics_snapshot: dict,
    understand_file_relname: str,
    original_relative_path: str,
):
    """
    Get lizard metrics using exact key, original path, and suffix fallback.
    """
    candidate_keys = [
        _normalize_path_for_keys(understand_file_relname),
        _normalize_path_for_keys(original_relative_path),
    ]

    for key in candidate_keys:
        if key in lizard_metrics_snapshot:
            return lizard_metrics_snapshot[key]

    normalized_original = _normalize_path_for_keys(original_relative_path)
    normalized_understand = _normalize_path_for_keys(understand_file_relname)

    for key, value in lizard_metrics_snapshot.items():
        if (
            key.endswith(normalized_original)
            or normalized_original.endswith(key)
            or key.endswith(normalized_understand)
            or normalized_understand.endswith(key)
        ):
            return value

    return []


def calculate_file_metrics_for_specific_files(
    understand_database,
    file_paths_string,
    lizard_metrics_snapshot=None,
):
    """
    Calculate file-level FAN-IN, FAN-OUT, and file-level complexity.
    """
    file_statistics = []
    any_files_missing = False

    if lizard_metrics_snapshot is None:
        lizard_metrics_snapshot = {}

    file_paths = _filter_java_files(_split_semicolon_paths(file_paths_string))

    if not file_paths:
        return file_statistics, any_files_missing

    for relative_path in file_paths:
        file_entity = _find_understand_file_entity(
            understand_database,
            relative_path,
        )

        if not file_entity:
            any_files_missing = True
            print(f"Understand file entity not found for file-level metrics: {relative_path}")
            continue

        lizard_functions = _get_lizard_functions_for_path(
            lizard_metrics_snapshot=lizard_metrics_snapshot,
            understand_file_relname=file_entity.relname(),
            original_relative_path=relative_path,
        )

        cyclomatic_values = [
            f.get("ccn", 0) or 0
            for f in lizard_functions
        ]

        cognitive_values = [
            f.get("cognitive_complexity", 0) or 0
            for f in lizard_functions
        ]

        method_details = [
            {
                "method_name": f.get("name"),
                "start_line": f.get("start_line"),
                "end_line": f.get("end_line"),
                "cyclomatic_complexity": f.get("ccn", 0) or 0,
                "cognitive_complexity": f.get("cognitive_complexity", 0) or 0,
            }
            for f in lizard_functions
        ]

        file_statistics.append(
            {
                "file_path": relative_path,
                "understand_file_relname": file_entity.relname(),
                "file_fan_in": len(file_entity.dependsby()) if file_entity.dependsby() else 0,
                "file_fan_out": len(file_entity.depends()) if file_entity.depends() else 0,
                "file_method_count": len(lizard_functions),
                "file_cyclomatic_complexity_sum": sum(cyclomatic_values),
                "file_cyclomatic_complexity_max": max(cyclomatic_values) if cyclomatic_values else 0,
                "file_cyclomatic_complexity_avg": (
                    sum(cyclomatic_values) / len(cyclomatic_values)
                    if cyclomatic_values else 0
                ),
                "file_cognitive_complexity_sum": sum(cognitive_values),
                "file_cognitive_complexity_max": max(cognitive_values) if cognitive_values else 0,
                "file_cognitive_complexity_avg": (
                    sum(cognitive_values) / len(cognitive_values)
                    if cognitive_values else 0
                ),
                "method_complexities": method_details,
            }
        )

    return file_statistics, any_files_missing


def calculate_metrics_for_specific_files(
    understand_database,
    file_paths_string,
    lizard_metrics_snapshot,
):
    """
    Calculate class-level FAN-IN, FAN-OUT, SLOC, cyclomatic complexity,
    and approximate cognitive complexity.
    """
    class_statistics = []
    any_files_missing = False

    file_paths = _filter_java_files(_split_semicolon_paths(file_paths_string))

    if not file_paths:
        return class_statistics, 0, any_files_missing

    for relative_path in file_paths:
        understand_entity = _find_understand_file_entity(
            understand_database,
            relative_path,
        )

        if not understand_entity:
            any_files_missing = True
            print(f"Understand file entity not found for class-level metrics: {relative_path}")
            continue

        lizard_functions = _get_lizard_functions_for_path(
            lizard_metrics_snapshot=lizard_metrics_snapshot,
            understand_file_relname=understand_entity.relname(),
            original_relative_path=relative_path,
        )

        print(
            f"{relative_path}: Understand relname={understand_entity.relname()}, "
            f"Lizard functions={len(lizard_functions)}"
        )

        class_entities = _get_classes_defined_in_file(
            understand_database,
            understand_entity,
        )

        if not class_entities:
            print(f"No class entities found in Understand for: {relative_path}")

        for class_entity in class_entities:
            try:
                if class_entity.library():
                    continue
            except Exception:
                pass

            total_cyclomatic_complexity = 0
            total_cognitive_complexity = 0
            method_complexity_details = []

            method_entities = _get_methods_defined_in_class(class_entity)

            if not method_entities:
                print(f"No method entities found for class: {class_entity.longname() or class_entity.name()}")

            for method_entity in method_entities:
                start_line_number = _get_entity_start_line(method_entity)

                if start_line_number is None:
                    continue

                matched_lizard_function = None

                for lizard_function in lizard_functions:
                    lizard_method_name = lizard_function.get("simple_name") or (
                        lizard_function["name"].split("::")[-1]
                    )

                    line_matches = (
                        lizard_function.get("start_line") is not None
                        and abs(lizard_function["start_line"] - start_line_number) <= 10
                    )

                    name_matches = lizard_method_name == method_entity.name()

                    if name_matches and line_matches:
                        matched_lizard_function = lizard_function
                        break

                if matched_lizard_function is None:
                    for lizard_function in lizard_functions:
                        if (
                            lizard_function.get("start_line") is not None
                            and abs(lizard_function["start_line"] - start_line_number) <= 3
                        ):
                            matched_lizard_function = lizard_function
                            break

                if matched_lizard_function is None:
                    continue

                method_cyclomatic = matched_lizard_function.get("ccn", 0) or 0
                method_cognitive = matched_lizard_function.get("cognitive_complexity", 0) or 0

                total_cyclomatic_complexity += method_cyclomatic
                total_cognitive_complexity += method_cognitive

                method_complexity_details.append(
                    {
                        "method_name": method_entity.longname() or method_entity.name(),
                        "start_line": start_line_number,
                        "end_line": matched_lizard_function.get("end_line"),
                        "cyclomatic_complexity": method_cyclomatic,
                        "cognitive_complexity": method_cognitive,
                    }
                )

            class_statistics.append(
                {
                    "class_name": class_entity.longname() or class_entity.name(),
                    "file_path": relative_path,
                    "understand_file_relname": understand_entity.relname(),
                    "fan_in": len(class_entity.dependsby()) if class_entity.dependsby() else 0,
                    "fan_out": len(class_entity.depends()) if class_entity.depends() else 0,
                    "sloc": class_entity.metric("CountLineCode") or 0,
                    "cyclomatic_complexity": total_cyclomatic_complexity,
                    "cognitive_complexity": total_cognitive_complexity,
                    "method_count_understand": len(method_entities),
                    "method_count_matched_lizard": len(method_complexity_details),
                    "method_complexities": method_complexity_details,
                }
            )

    return class_statistics, len(class_statistics), any_files_missing


def analyze_and_extract_understand_metrics(
    source_root_directory,
    understand_db_path,
    project_key,
    affected_files_string,
    lizard_metrics_snapshot,
    compute_class_metrics=True,
    compute_file_metrics=True,
):
    """
    Analyze repository using Understand and extract metrics for selected files.
    """
    _ensure_understand_project_analyzed(
        source_root_directory,
        understand_db_path,
        project_key,
    )

    understand_database = understand.open(understand_db_path) if understand else None

    if understand_database is None:
        raise RuntimeError("Failed to open Understand database.")

    class_details_json = pd.NA
    file_details_json = pd.NA
    status = "Success"
    missing_any = False

    try:
        if compute_class_metrics:
            class_statistics, _, missing_class_files = calculate_metrics_for_specific_files(
                understand_database,
                affected_files_string,
                lizard_metrics_snapshot,
            )

            class_details_json = json.dumps(class_statistics) if class_statistics else pd.NA
            missing_any |= missing_class_files

        if compute_file_metrics:
            file_statistics, missing_file_files = calculate_file_metrics_for_specific_files(
                understand_database,
                affected_files_string,
                lizard_metrics_snapshot,
            )

            file_details_json = json.dumps(file_statistics) if file_statistics else pd.NA
            missing_any |= missing_file_files

        if missing_any:
            status = "Success (Partial: Some files missing)"

    finally:
        understand_database.close()

    return {
        "ClassDetails_JSON": class_details_json,
        "FileDetails_JSON": file_details_json,
        "Status": status,
    }


# ============================================================
# Git diff metrics
# ============================================================

def get_file_changes_between_commits(
    repository_path,
    base_commit_hash,
    comparison_commit_hash,
    path_map: dict[str, Optional[str]],
):
    """
    Calculate Git diff metrics between intro and payment commits.
    """
    if (
        not base_commit_hash
        or not comparison_commit_hash
        or base_commit_hash == comparison_commit_hash
    ):
        return {
            "TotalFilesChangedInScope": 0,
            "TotalLinesAddedInScope": 0,
            "TotalLinesDeletedInScope": 0,
            "FileChangesDetails_JSON": json.dumps([]),
            "FileChangesStatus": "Skipped",
        }

    if not path_map:
        return {
            "TotalFilesChangedInScope": 0,
            "TotalLinesAddedInScope": 0,
            "TotalLinesDeletedInScope": 0,
            "FileChangesDetails_JSON": json.dumps([]),
            "FileChangesStatus": "Success (Empty File Scope)",
        }

    file_change_details = []
    total_lines_added = 0
    total_lines_deleted = 0
    total_files_changed = 0

    seen_numstat_paths = set()

    path_pairs = list(path_map.items())

    for i in range(0, len(path_pairs), 25):
        batch_pairs = path_pairs[i: i + 25]

        batch_paths = []
        seen_batch_paths = set()

        for intro_path, payment_path in batch_pairs:
            candidate_paths = [intro_path]

            if payment_path and payment_path != intro_path:
                candidate_paths.append(payment_path)

            for p in candidate_paths:
                if not p:
                    continue

                git_path = p.replace("\\", "/")
                normalized = _normalize_path_for_keys(git_path)

                if normalized not in seen_batch_paths:
                    batch_paths.append(git_path)
                    seen_batch_paths.add(normalized)

        if not batch_paths:
            continue

        command = [
            GIT_EXECUTABLE_PATH,
            "-C",
            repository_path,
            "diff",
            "--numstat",
            "--find-renames=20%",
            base_commit_hash,
            comparison_commit_hash,
            "--",
        ] + batch_paths

        command_result = _execute_command(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )

        if command_result.returncode != 0:
            print("Warning: git diff --numstat failed.")
            print("STDOUT:")
            print(command_result.stdout)
            print("STDERR:")
            print(command_result.stderr)
            continue

        for line in command_result.stdout.strip().splitlines():
            parts = line.split("\t")

            if len(parts) < 3:
                continue

            added_lines_str = parts[0]
            deleted_lines_str = parts[1]
            git_numstat_path = "\t".join(parts[2:]).replace("\\", "/")

            if git_numstat_path in seen_numstat_paths:
                continue

            seen_numstat_paths.add(git_numstat_path)

            added_lines_int = int(added_lines_str) if added_lines_str.isdigit() else 0
            deleted_lines_int = int(deleted_lines_str) if deleted_lines_str.isdigit() else 0

            file_change_details.append(
                {
                    "git_numstat_path": git_numstat_path,
                    "lines_added": added_lines_int,
                    "lines_deleted": deleted_lines_int,
                }
            )

            total_lines_added += added_lines_int
            total_lines_deleted += deleted_lines_int
            total_files_changed += 1

    return {
        "TotalFilesChangedInScope": total_files_changed,
        "TotalLinesAddedInScope": total_lines_added,
        "TotalLinesDeletedInScope": total_lines_deleted,
        "FileChangesDetails_JSON": json.dumps(file_change_details),
        "FileChangesStatus": "Success",
    }


# ============================================================
# Commit-level metrics
# ============================================================

def get_parent_commit_hash(repository_path, commit_hash):
    """
    Get parent commit hash.
    """
    try:
        return run_git_command(repository_path, ["rev-parse", f"{commit_hash}^1"])
    except Exception:
        return None


def get_deleted_files_count_in_commit_vs_parent(repository_path, commit_hash):
    """
    Count deleted files in a commit compared to its parent.
    """
    parent_commit_hash = get_parent_commit_hash(repository_path, commit_hash)

    if parent_commit_hash is None:
        return 0

    diff = run_git_command(
        repository_path,
        ["diff", "--name-only", "--diff-filter=D", parent_commit_hash, commit_hash],
    )

    return len(diff.splitlines()) if diff else 0


def get_total_files_in_repo_at_commit(repository_path, commit_hash):
    """
    Count total tracked files in repo at a specific commit.
    """
    ls_files_output = run_git_command(
        repository_path,
        ["ls-tree", "-r", "--name-only", commit_hash],
    )

    return len(ls_files_output.splitlines()) if ls_files_output else 0


# ============================================================
# Main processing
# ============================================================

def _get_row_value(row, column_name: str, default: str = ""):
    """
    Safe row value getter.
    """
    if column_name not in row:
        return default

    value = row.get(column_name, default)

    if pd.isna(value):
        return default

    return str(value).strip()


def process_csv_file(input_file_path, output_file_path):
    """
    Process CSV and extract metrics.
    """
    print("Loading CSVs...")

    dataframe = None

    if os.path.exists(output_file_path):
        try:
            dataframe = pd.read_csv(output_file_path, dtype=str).fillna(pd.NA)
            print("Resuming from previous output file...")
        except Exception:
            dataframe = None

    if dataframe is None:
        dataframe = pd.read_csv(input_file_path, dtype=str).fillna(pd.NA)

    if "Key" not in dataframe.columns:
        raise RuntimeError("Input CSV must contain a 'Key' column, e.g., LUCENE-1234.")

    print("Project keys:")
    print(dataframe["Key"].astype(str).str.split("-").str[0].value_counts())

    if dataframe.empty:
        print("No data to process...")
        return

    for prefix in ["Intro", "Payment"]:
        for suffix in understand_metric_suffixes:
            column_name = f"{prefix}_{suffix}"

            if column_name not in dataframe.columns:
                dataframe[column_name] = pd.NA

        for suffix in new_commit_level_metric_suffixes:
            column_name = f"{prefix}_{suffix}"

            if column_name not in dataframe.columns:
                dataframe[column_name] = pd.NA

    for column_name in git_diff_metric_column_names + custom_added_columns:
        if column_name not in dataframe.columns:
            dataframe[column_name] = pd.NA

    dataframe["_ProjectKey"] = (
        dataframe["Key"]
        .astype(str)
        .str.split("-")
        .str[0]
        .str.upper()
        .str.strip()
    )

    grouped = dataframe.groupby("_ProjectKey", dropna=False)

    for project_key, row_indices in tqdm(grouped.groups.items(), desc="Repositories"):
        project_key = str(project_key).upper().strip()

        if not project_key or project_key.lower() in ["nan", "<na>", "none"]:
            continue

        try:
            repository_path = get_repository_path_from_project_key(project_key)
        except Exception as e:
            print(f"Cannot resolve repository path for {project_key}: {e}")
            continue

        if not os.path.isdir(repository_path):
            print(f"Repository path not found for {project_key}: {repository_path}")
            continue

        restore_git_state(repository_path, project_key, "Group start")

        print(
            f"Processing repository {os.path.basename(repository_path)} "
            f"({len(row_indices)} rows)..."
        )

        for row_index in tqdm(list(row_indices), desc="Rows", leave=False):
            current_row = dataframe.loc[row_index]
            issue_key = _get_row_value(current_row, "Key", "N/A_KEY")

            raw_introduction_files = _get_row_value(
                current_row,
                "Intro_Affected_Files",
                "",
            )

            introduction_files_list = sorted(set(_split_semicolon_paths(raw_introduction_files)))
            java_introduction_files_list = _filter_java_files(introduction_files_list)

            if not java_introduction_files_list:
                print(f"{issue_key} Skipping row (no Java intro files)...")
                continue

            introduction_commit_hash = _get_row_value(current_row, "Intro Hash", "")
            payment_commit_hash = _get_row_value(current_row, "Payment Hash", "")

            print(f"\n=== Processing {issue_key} ===")
            print(f"Repository  : {repository_path}")
            print(f"Intro hash  : {introduction_commit_hash}")
            print(f"Payment hash: {payment_commit_hash}")
            print("Java intro files:")
            for p in java_introduction_files_list:
                print(f"  {p}")

            # ============================================================
            # 1. Intro metrics
            # ============================================================

            intro_class_metrics_extracted = _is_json_content_present(
                current_row["Intro_ClassDetails_JSON"]
            )

            intro_file_metrics_extracted = _is_json_content_present(
                current_row["Intro_FileDetails_JSON"]
            )

            if intro_class_metrics_extracted:
                print(f"{issue_key} Intro class details present, skipping class extraction...")

            if intro_file_metrics_extracted:
                print(f"{issue_key} Intro file details present, skipping file extraction...")

            if not (intro_class_metrics_extracted and intro_file_metrics_extracted):
                if _is_valid_commit_hash(introduction_commit_hash):
                    print(
                        f"{issue_key} Refreshing intro metrics "
                        f"(class={not intro_class_metrics_extracted}, "
                        f"file={not intro_file_metrics_extracted})..."
                    )

                    temp_intro_directory = tempfile.mkdtemp(
                        prefix=f"und_{issue_key}_intro_",
                        dir=CUSTOM_TMPDIR,
                    )

                    understand_intro_db_path = os.path.join(
                        temp_intro_directory,
                        "project.und",
                    )

                    try:
                        checkout_commit_safely(
                            repository_path=repository_path,
                            commit_hash=introduction_commit_hash,
                            context=f"{issue_key} intro",
                        )

                        print_debug_head_and_files(
                            repository_path=repository_path,
                            file_paths=introduction_files_list,
                            context=f"{issue_key} intro",
                        )

                        existing_intro_java_files = verify_files_exist_in_worktree(
                            repository_path=repository_path,
                            file_paths=java_introduction_files_list,
                            context=f"{issue_key} intro",
                        )

                        if not existing_intro_java_files:
                            raise RuntimeError(
                                f"{issue_key} No Java intro files exist after checkout to "
                                f"{introduction_commit_hash}"
                            )

                        lizard_intro_metrics = get_lizard_metrics_with_api(
                            repository_path,
                            existing_intro_java_files,
                        )

                        introduction_metrics = analyze_and_extract_understand_metrics(
                            source_root_directory=repository_path,
                            understand_db_path=understand_intro_db_path,
                            project_key=project_key,
                            affected_files_string=";".join(existing_intro_java_files),
                            lizard_metrics_snapshot=lizard_intro_metrics,
                            compute_class_metrics=not intro_class_metrics_extracted,
                            compute_file_metrics=not intro_file_metrics_extracted,
                        )

                        if not intro_class_metrics_extracted:
                            dataframe.at[row_index, "Intro_ClassDetails_JSON"] = (
                                introduction_metrics["ClassDetails_JSON"]
                            )

                        if not intro_file_metrics_extracted:
                            dataframe.at[row_index, "Intro_FileDetails_JSON"] = (
                                introduction_metrics["FileDetails_JSON"]
                            )

                        if "success" not in str(current_row["Intro_Status"]).lower():
                            dataframe.at[row_index, "Intro_Status"] = (
                                introduction_metrics["Status"]
                            )

                    except Exception as e:
                        print(f"{issue_key} Intro metrics failed: {e}")
                        dataframe.at[row_index, "Intro_Status"] = f"Failed: {e}"

                    finally:
                        shutil.rmtree(temp_intro_directory, ignore_errors=True)
                        _clear_scitools_db_cache_on_linux(max_gb=2.0)

                else:
                    dataframe.at[row_index, "Intro_Status"] = "Skipped (Invalid Intro Hash)"

            # ============================================================
            # 2. Map intro paths to payment paths
            # ============================================================

            path_map: dict[str, Optional[str]] = {}
            payment_paths: list[str] = []
            missing_intro_files: list[str] = []

            if (
                _is_valid_commit_hash(introduction_commit_hash)
                and _is_valid_commit_hash(payment_commit_hash)
            ):
                try:
                    checkout_commit_safely(
                        repository_path=repository_path,
                        commit_hash=introduction_commit_hash,
                        context=f"{issue_key} before path mapping",
                    )

                    path_map, payment_paths, missing_intro_files = (
                        map_intro_paths_to_payment_paths(
                            repository_path=repository_path,
                            introduction_files_list=introduction_files_list,
                            introduction_commit_hash=introduction_commit_hash,
                            payment_commit_hash=payment_commit_hash,
                        )
                    )

                    dataframe.at[row_index, "Payment_Path_Map_JSON"] = json.dumps(path_map)

                    if missing_intro_files:
                        dataframe.at[row_index, "Payment_MissingIntroFiles"] = (
                            ";".join(missing_intro_files)
                        )
                    else:
                        dataframe.at[row_index, "Payment_MissingIntroFiles"] = pd.NA

                except Exception as e:
                    print(f"{issue_key} Path mapping failed: {e}")
                    dataframe.at[row_index, "Payment_Path_Map_JSON"] = json.dumps({})
                    dataframe.at[row_index, "Payment_MissingIntroFiles"] = pd.NA

            else:
                print(f"{issue_key} Invalid intro/payment hash, cannot map paths...")
                dataframe.at[row_index, "Payment_Path_Map_JSON"] = json.dumps({})
                dataframe.at[row_index, "Payment_MissingIntroFiles"] = pd.NA

            # ============================================================
            # 3. Payment metrics
            # ============================================================

            payment_class_metrics_extracted = _is_json_content_present(
                current_row["Payment_ClassDetails_JSON"]
            )

            payment_file_metrics_extracted = _is_json_content_present(
                current_row["Payment_FileDetails_JSON"]
            )

            if payment_class_metrics_extracted:
                print(f"{issue_key} Payment class details present, skipping class extraction...")

            if payment_file_metrics_extracted:
                print(f"{issue_key} Payment file details present, skipping file extraction...")

            if not (payment_class_metrics_extracted and payment_file_metrics_extracted):
                if _is_valid_commit_hash(payment_commit_hash):
                    print(
                        f"{issue_key} Refreshing payment metrics "
                        f"(class={not payment_class_metrics_extracted}, "
                        f"file={not payment_file_metrics_extracted})..."
                    )

                    temp_payment_directory = tempfile.mkdtemp(
                        prefix=f"und_{issue_key}_payment_",
                        dir=CUSTOM_TMPDIR,
                    )

                    understand_payment_db_path = os.path.join(
                        temp_payment_directory,
                        "project.und",
                    )

                    try:
                        checkout_commit_safely(
                            repository_path=repository_path,
                            commit_hash=payment_commit_hash,
                            context=f"{issue_key} payment",
                        )

                        java_payment_paths = _filter_java_files(payment_paths)

                        print_debug_head_and_files(
                            repository_path=repository_path,
                            file_paths=java_payment_paths,
                            context=f"{issue_key} payment",
                        )

                        existing_payment_java_files = verify_files_exist_in_worktree(
                            repository_path=repository_path,
                            file_paths=java_payment_paths,
                            context=f"{issue_key} payment",
                        )

                        if existing_payment_java_files:
                            lizard_payment_metrics = get_lizard_metrics_with_api(
                                repository_path,
                                existing_payment_java_files,
                            )

                            payment_metrics = analyze_and_extract_understand_metrics(
                                source_root_directory=repository_path,
                                understand_db_path=understand_payment_db_path,
                                project_key=project_key,
                                affected_files_string=";".join(existing_payment_java_files),
                                lizard_metrics_snapshot=lizard_payment_metrics,
                                compute_class_metrics=not payment_class_metrics_extracted,
                                compute_file_metrics=not payment_file_metrics_extracted,
                            )

                            if not payment_class_metrics_extracted:
                                dataframe.at[row_index, "Payment_ClassDetails_JSON"] = (
                                    payment_metrics["ClassDetails_JSON"]
                                )

                            if not payment_file_metrics_extracted:
                                dataframe.at[row_index, "Payment_FileDetails_JSON"] = (
                                    payment_metrics["FileDetails_JSON"]
                                )

                            if "success" not in str(current_row["Payment_Status"]).lower():
                                dataframe.at[row_index, "Payment_Status"] = (
                                    payment_metrics["Status"]
                                )

                        else:
                            print(
                                f"{issue_key} No mapped Java intro files exist at payment hash, "
                                f"skipping payment UDB metrics..."
                            )

                            dataframe.at[row_index, "Payment_Status"] = (
                                "Skipped (No mapped Java intro files exist at payment hash)"
                            )

                    except Exception as e:
                        print(f"{issue_key} Payment metrics failed: {e}")
                        dataframe.at[row_index, "Payment_Status"] = f"Failed: {e}"

                    finally:
                        shutil.rmtree(temp_payment_directory, ignore_errors=True)
                        _clear_scitools_db_cache_on_linux(max_gb=2.0)

                else:
                    dataframe.at[row_index, "Payment_Status"] = "Skipped (Invalid Payment Hash)"

            # ============================================================
            # 4. Git diff metrics
            # ============================================================

            git_diff_already_calculated = _is_json_content_present(
                current_row["Payment_FileChangesDetails_JSON"]
            )

            if git_diff_already_calculated:
                print(f"{issue_key} Git diff already present, skipping Git diff...")

            if not git_diff_already_calculated:
                print(f"{issue_key} Calculating Git diff for intro→payment...")

                try:
                    git_diff_metrics = get_file_changes_between_commits(
                        repository_path=repository_path,
                        base_commit_hash=introduction_commit_hash,
                        comparison_commit_hash=payment_commit_hash,
                        path_map=path_map,
                    )

                    for metric_key, metric_value in git_diff_metrics.items():
                        dataframe.at[row_index, f"Payment_{metric_key}"] = metric_value

                except Exception as e:
                    print(f"{issue_key} Git diff failed: {e}")
                    dataframe.at[row_index, "Payment_FileChangesStatus"] = f"Failed: {e}"

            dataframe.to_csv(output_file_path, index=False)

        restore_git_state(repository_path, project_key, "Group end")
        dataframe.to_csv(output_file_path, index=False)

    if "_ProjectKey" in dataframe.columns:
        dataframe = dataframe.drop(columns=["_ProjectKey"])

    dataframe.to_csv(output_file_path, index=False)

    print("Processing complete...")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Script start...")

    os.makedirs(DATA_DIRECTORY, exist_ok=True)

    if os.path.dirname(METRICS_DATASET_OUTPUT_PATH):
        os.makedirs(os.path.dirname(METRICS_DATASET_OUTPUT_PATH), exist_ok=True)

    os.makedirs(CUSTOM_TMPDIR, exist_ok=True)

    if not os.path.isdir(REPOS_ROOT_DIRECTORY):
        print(f"Repository root directory not found: {REPOS_ROOT_DIRECTORY}")
        sys.exit(1)

    print(f"Repository root directory: {REPOS_ROOT_DIRECTORY}")

    setup_git_executable()

    if not setup_understand_environment():
        print("Understand setup failed. Exiting...")
        sys.exit(1)

    try:
        process_csv_file(
            TRACED_DATASET_INPUT_PATH,
            METRICS_DATASET_OUTPUT_PATH,
        )

    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print("Done...")