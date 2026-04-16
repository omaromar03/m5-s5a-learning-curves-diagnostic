# m5-s5a-learning-curves-diagnostic

##Learning Curve Analysis

I used the F1 metric as an evaluation criterion instead of accuracy because the telecom customer outage dataset is unbalanced, and the F1 metric provides a better balance between accuracy and recall for the underrepresented customer segment. From the learning curve, we observe that the training score starts relatively high and then decreases as the training set size increases, while the validation score rises and then plateaus. Importantly, the gap between the training and validation curves narrows as the data size increases, indicating that the model does not suffer from high variance.

However, the training and validation scores converge to relatively low values ​​(around 0.35–0.37), suggesting that the model suffers from high bias (lack of concordance). This indicates that the logistic regression model is too simplistic to capture the underlying patterns in the data. Furthermore, since the validation curve plateaus and does not improve significantly with increasing data size, collecting additional data is unlikely to result in substantial performance improvements.

Therefore, the most appropriate next step is to increase the model's complexity. This can be achieved by using more flexible models such as random forests or gradient enhancement, or by introducing more expressive properties (such as interaction boundaries or nonlinear transformations). These approaches are more likely to be able to detect complex relationships in the data and improve overall performance.
## What this project includes
- `learning_curves_diagnostic.py` — script to compute learning curves and save the plot
- `learning_curve.png` — plot of training vs validation F1-score
- Written analysis connecting the learning curves to bias-variance concepts

## Method
- Logistic Regression
- Stratified 5-fold cross-validation
- At least 5 training set sizes
- Training and validation curves with ±1 standard deviation confidence bands
- F1-score as the evaluation metric

## How to run

```bash
pip install -r requirements.txt
python learning_curves_diagnostic.py