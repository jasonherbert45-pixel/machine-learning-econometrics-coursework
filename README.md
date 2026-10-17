
# Machine Learning & Econometric Analysis

An exploratory machine learning project completed as group coursework at the University of Bath, investigating the behaviour, predictive performance, and limitations of various statistical and machine learning techniques.

## Project Overview

This project was completed for the Advanced Econometrics and Data Analytics module during the 2025–2026 academic year.

The analysis consists of two parts:

### Part 1: Regression and Model Selection

An investigation into regression techniques and the trade-off between model complexity and generalisation.

Methods examined:
- Ordinary Least Squares (OLS)
- Ridge and Lasso regression
- Decision trees and pruning
- Bagging and random forests
- Gradient boosting

Models were evaluated using training and test mean squared error, cross-validation, and repeated random splits.

### Part 2: Income Classification

An exploratory classification analysis examining relationships between demographic and socioeconomic characteristics and income categories.

Methods examined include:
- Logistic regression
- Linear and Quadratic Discriminant Analysis
- Naive Bayes
- Tree-based and ensemble methods
- Neural networks

The analysis compares classification performance, investigates variable importance, and examines how model behaviour changes under alternative training/test splits.

## Key Findings

- Regularisation methods improved out-of-sample performance relative to OLS in the first investigation.
- Different model families illustrated trade-offs between flexibility, complexity, and generalisation.
- Income classification results highlighted the importance of several demographic and socioeconomic variables.
- Comparing models helped identify both useful patterns and limitations in interpreting machine learning results.

For detailed results, methodology, figures, and discussion, see the full coursework report.

## Technologies

- R
- RStudio
- Statistical modelling and machine learning
- Cross-validation and model evaluation
- Data visualisation

## Repository Contents

- `ES32029 ML Coursework R Script.r` — R analysis for both parts
- `ES32029 ML Coursework.pdf` — Full academic report
- `data.sim.34.RData` — Part 1 dataset
- `data.34.RData` — Part 2 dataset

## Reproducing the Analysis

The analysis was written in R.

The R script installs required packages if they are missing and loads the datasets from the working directory.

To reproduce the analysis:
1. Open the project directory in RStudio.
2. Set the working directory to the repository root.
3. Open the R script.
4. Run the relevant analysis sections.

The original script includes configurable random seeds and training/test split proportions to reproduce the coursework comparisons.

Reproducibility has not yet been independently verified in a clean R environment.

## Academic Attribution

This project was completed collaboratively by Group 34 for the Advanced Econometrics and Data Analytics module at the University of Bath (2025–2026).

The report and code represent collaborative academic coursework.

Published with the permission of the course lecturer and the agreement of the group members.

This is a student portfolio project and does not represent an official publication or endorsement by the University of Bath.

## Contributors

This project was completed collaboratively by Group 34 at the University of Bath:

- Jack Guest
- Sam Harris
- Danny Fowler
- Harry Solomon
- Jason Herbert

## Reproducibility testing — October 2026

The original analysis was rerun using R 4.6.1 on Windows 11.
Both parts of the coursework executed successfully. For Part 2, all five random seeds were tested using a 30% test split.
Of the 30 classification results reported in the coursework, 27 matched to three decimal places. Three gradient boosting results differed slightly. Repeated execution confirmed that the Seed 5 gradient boosting result was stable in the tested environment.
The complete R environment is documented in session-info.txt.