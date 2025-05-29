# AI/ML Internship - Task 3: Linear Regression

## Task Overview
Implemented simple and multiple linear regression using scikit-learn on the California Housing dataset.

## Files
- `linear_regression_task.py`: Main Python script
- `regression_plot.png`: Output visualization

## How to Run
1. Install requirements: `pip install pandas scikit-learn matplotlib`
2. Run script: `python linear_regression_task.py`

## Key Learnings
- Learned how to format data for sklearn
- Understood the difference between MAE and MSE
- Discovered that Boston dataset is deprecated (oops!)
- Realized importance of random_state in train_test_split

## Mistakes Made
1. Initially tried to use Boston Housing dataset (deprecated)
2. Forgot to save the plot at first
3. Took several tries to get the coefficient printing right
4. Struggled with single vs double brackets for features

## Answers to Interview Questions
1. **Assumptions**: Linearity, independence, homoscedasticity, normal residuals
2. **Coefficients**: Change in y per unit change in x, holding others constant
3. **R²**: Proportion of variance explained (0-1)
4. **MSE vs MAE**: MSE penalizes outliers more heavily
5. **Multicollinearity**: High VIF or correlation matrix
6. **Simple vs Multiple**: One vs multiple predictors
7. **Classification**: Not directly, but can be thresholded
8. **Violations**: Biased estimates, poor predictions
