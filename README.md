# Replication package of Does It Matter Who Repays Architecture Technical Debt? An Empirical Study of Dependency and Complexity Changes

## Description of this study:
**Background**: Architecture Technical Debt (ATD) refers to sub-optimal design decisions that can slow down software maintenance and evolution. Although prior studies have examined identifying ATD, as well as its causes and impacts, less is known about what happens structurally when ATD is repaid. In particular, limited evidence exists on whether self-fixed ATD, i.e. ATD repaid by the developer who introduced it in the first place, and non--self-fixed ATD, repaid by another developer, exhibit different structural evolution patterns.

**Aims**: This study investigates how dependency-related metrics, complexity-related metrics, and file-level change characteristics evolve from ATD introduction to repayment. Specifically, we examine whether self-fixed and non--self-fixed ATD differ in terms of changes in Fan-In, Fan-Out, cyclomatic complexity, cognitive complexity, and the relationship between file change frequency and structural metric changes.

**Method**: We analyze ATD items mined from issue trackers and linked to both introduction and repayment commits. Each item is classified as self-fixed when the original introducer also performs the repayment, and as non--self-fixed otherwise. We measure Fan-In and Fan-Out, cyclomatic and cognitive complexity to capture dependency- and complexity-related changes. 

**Results**: The findings show that ATD repayment is associated with measurable but uneven structural changes. Dependency-related evidence is stronger at the issue level, particularly for Fan-In in non--self-fixed ATD, whereas file-level dependency changes are statistically significant but practically marginal. Complexity-related changes are mainly observed for non--self-fixed cyclomatic complexity, although the effect sizes are small or negligible. File change frequency is consistently associated with changes in dependency and complexity metrics, especially Fan-Out and cyclomatic complexity.

**Conclusions**: ATD repayment may reshape affected source code rather than simply remove architectural debt. The results indicate that non--self-fixed ATD can be associated with greater structural change, and that repeated file modifications are linked to changes in dependencies and complexity. 

## Contents

This replication package includes the dataset, analysis scripts, and figures used in the empirical study of self-fixed and non-self-fixed Architecture Technical Debt (ATD).

## Directory Overview

| Path | Description |
|---|---|
| `code/` | Contains Python scripts for metric extraction, statistical analysis, and visualization. |
| `dataset/` | Contains the traced ATD dataset used in the study. |
| `figures/` | Contains the generated figures reported in the paper. |

## Code Files

| File | Description |
|---|---|
| `understand_dependency_complexity_extraction.py` | Extracts dependency and complexity metrics from the analyzed source-code projects using Understand by SciTools. The extracted metrics include dependency-related measures such as Fan-In and Fan-Out, as well as complexity-related measures used in the subsequent RQ1 and RQ2 analyses. |
| `intro-payment-metrics.py` | Prepares metric values for the ATD introduction and repayment phases. |
| `rq1-rq2-wilcoxon-dan-boxplot-issue-level.py` | Performs issue-level Wilcoxon signed-rank tests and generates boxplots for RQ1 and RQ2. |
| `rq1_rq2_mann_whitney_and_boxplot.py` | Performs Mann–Whitney U tests to compare self-fixed and non-self-fixed ATD items. |
| `rq3-file-change-analysis.py` | Analyzes the relationship between file change frequency and metric deltas for RQ3. |
| `requirements.txt` | Lists the Python dependencies required to run the replication package scripts. |

## Dataset Files

| File | Description |
|---|---|
| `dataset/atd-dataset.csv` | Contains the ATD dataset collected from Jira issue trackers. |
| `dataset/ATD-SELF-FIXED-FINAL-DATASET-TRACED.csv` | Final traced ATD dataset containing ATD items, repayment classification, lifecycle commits, affected files, and associated metrics. |

## Figure Files

| File | Description |
|---|---|
| `RQ1_dependency_metrics_file_level.pdf` | File-level visualization of dependency metric changes for RQ1. |
| `RQ1_dependency_metrics.pdf` | Issue-level visualization of dependency metric changes for RQ1. |
| `RQ2_complexity_metrics_file_level.pdf` | File-level visualization of complexity metric changes for RQ2. |
| `RQ2_complexity_metrics.pdf` | Issue-level visualization of complexity metric changes for RQ2. |
| `rq3_complexity_file_commit_count_vs_metric_deltas.pdf` | Scatter plots showing the relationship between file change frequency and complexity metric deltas for RQ3. |
| `rq3_dependency_file_commit_count_vs_metric_deltas.pdf` | Scatter plots showing the relationship between file change frequency and dependency metric deltas for RQ3. |


## Reproducing the Metric Extraction

The script `understand_dependency_complexity_extraction.py` is used to extract dependency and complexity metrics from the analyzed projects. It relies on **Understand by SciTools** to automatically analyze the source code at two lifecycle points for each traced ATD issue: the **introduction commit** and the **payment commit**.

The script uses the traced issue dataset:

```text
dataset/ATD-SELF-FIXED-FINAL-DATASET-TRACED.csv
```

These paths are defined as global variables in `understand_dependency_complexity_extraction.py`. If Understand is installed in a different location, edit the corresponding path variable in the script.

For Windows, update the Understand path as follows:

```python
UNDERSTAND_BIN_WINDOWS = r"C:\Program Files\SciTools\bin\pc-win64"
```

For Linux, update the Understand path as follows:
```python
UNDERSTAND_BIN_LINUX = "/root/scitools/bin/linux64"
```

If your Understand installation is located elsewhere, replace these values with the correct local installation path.

### Environment and Requirements

The replication package was prepared and tested with the following environment:

| Component | Version / File |
|---|---|
| Python | Python 3.13.7 |
| Understand by SciTools | Build 1236 |
| Python dependencies | `requirements.txt` |

Install the required Python packages from the root directory of the replication package using:

```bash
pip install -r requirements.txt
```

### Running Python Scripts

All Python scripts can be executed from the root directory of the replication package using the following general command:

```bash
python "code/<script_name>.py"

```
