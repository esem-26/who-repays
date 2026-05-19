# Replication package of Does It Matter Who Repays Architecture Technical Debt? An Empirical Study of Dependency and Complexity Changes

## Description of this study:
**Background**: Architecture Technical Debt (ATD) refers to sub-optimal design decisions that can slow down software maintenance and evolution. Although prior studies have examined identifying ATD, as well as its causes and impacts, less is known about what happens structurally when ATD is repaid. In particular, limited evidence exists on whether self-fixed ATD, i.e. ATD repaid by the developer who introduced it in the first place, and non--self-fixed ATD, repaid by another developer, exhibit different structural evolution patterns.

**Aims**: This study investigates how dependency-related metrics, complexity-related metrics, and file-level change characteristics evolve from ATD introduction to repayment. Specifically, we examine whether self-fixed and non--self-fixed ATD differ in terms of changes in Fan-In, Fan-Out, cyclomatic complexity, cognitive complexity, and the relationship between file change frequency and structural metric changes.

**Method**: We analyze ATD items mined from issue trackers and linked to both introduction and repayment commits. Each item is classified as self-fixed when the original introducer also performs the repayment, and as non--self-fixed otherwise. We measure Fan-In and Fan-Out, cyclomatic and cognitive complexity to capture dependency- and complexity-related changes. 

**Results**: The findings show that ATD repayment is associated with measurable but uneven structural changes. Dependency-related evidence is stronger at the issue level, particularly for Fan-In in non--self-fixed ATD, whereas file-level dependency changes are statistically significant but practically marginal. Complexity-related changes are mainly observed for non--self-fixed cyclomatic complexity, although the effect sizes are small or negligible. File change frequency is consistently associated with changes in dependency and complexity metrics, especially Fan-Out and cyclomatic complexity.

**Conclusions**: ATD repayment may reshape affected source code rather than simply remove architectural debt. The results indicate that non--self-fixed ATD can be associated with greater structural change, and that repeated file modifications are linked to changes in dependencies and complexity. 

## Contents
