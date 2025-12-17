Overview :

This project implements an end-to-end credit risk default prediction pipeline using supervised machine learning. The objective is to predict the probability of loan default by leveraging internal bank data and external credit bureau data, while carefully handling real-world constraints such as missing values, feature inconsistency, and unseen data validation. The workflow reflects practical banking and risk analytics scenarios, emphasizing data integrity, leakage prevention, and conservative model evaluation.

Problem Statement:

Accurate credit risk assessment is critical for financial institutions to reduce default losses while maintaining efficient lending operations. Traditional rule-based approaches often fail to capture complex, non-linear relationships in borrower behavior. The goal of this project is to build a machine learning model to classify borrowers as defaulters or non-defaulters based on historical credit and transaction data, with an emphasis on robustness and generalization.

Dataset Description:

The project uses three datasets. The internal bank dataset contains loan-level and customer transaction attributes. The external credit bureau dataset includes credit history variables such as repayment behavior, delinquencies, and bureau-derived risk indicators. The unseen dataset is a hold-out dataset intended to simulate real-world deployment and validate generalization performance. All datasets are merged using unique customer identifiers after preprocessing.

Approach:


Data Preprocessing:

Missing values were identified and handled using selective, feature-aware imputation rather than blanket filling. Duplicate and inconsistent records were removed. Outliers in skewed financial variables were treated to stabilize distributions. Categorical variables were encoded using appropriate encoding techniques. Strict separation between training data and unseen data was maintained throughout to prevent data leakage.

Feature Engineering:

Credit utilization ratios and repayment behavior indicators were derived. Delinquency-related variables were aggregated to capture historical risk patterns. Numerical features were normalized to improve training stability. Feature pruning was performed to reduce multicollinearity and noise.

Model Development:

Multiple models were explored to balance interpretability and predictive performance. Logistic regression was used as an interpretable baseline. Tree-based models were evaluated to capture non-linear feature interactions. Hyperparameter tuning was performed using cross-validation.

Model Evaluation:

Model performance was evaluated using accuracy, precision, recall, F1-score, and ROC-AUC. Evaluation was conducted conservatively, with careful consideration of feature availability in the unseen dataset to avoid misleading conclusions.

Challenges Faced and Design Decisions:


Inconsistent Feature Availability in Unseen Data:

The unseen validation dataset did not contain critical predictive features such as credit score, which were present in the training data. This resulted in a schema mismatch that prevented direct inference. As a result, the unseen dataset was initially used only for distributional analysis, such as frequency histograms, rather than prediction. Model versions that assumed full feature availability explicitly excluded unseen data from scoring to maintain methodological correctness.

Multiple Feature and Model Iterations:

The project involved multiple modeling iterations due to evolving feature representations. These included a baseline model using 43 core variables, expanded versions with one-hot encoded categorical features, and refined versions after feature pruning and transformation. Each iteration was clearly documented in the notebook using structured headings to ensure traceability and reproducibility.

Limited Use of Unseen Data in Early Versions:

Due to feature inconsistency, the first model version used the unseen dataset strictly for exploratory analysis rather than prediction or scoring. Using unseen data for inference without aligned features would have produced misleading performance metrics. Excluding it from early prediction stages was a deliberate and methodologically sound decision.

Data Quality Issues During EDA and Cleaning:

A large portion of the dataset contained missing or incomplete values, particularly in bureau-related and financial fields. This complicated exploratory data analysis and increased the risk of biased imputation. To mitigate this, missing value analysis was performed to identify high-risk variables, selective imputation strategies were applied, and features with excessive sparsity were removed to improve model stability.

Results:

The models demonstrated strong discriminatory power between defaulters and non-defaulters. Tree-based models outperformed linear baselines in capturing complex feature interactions. Conservative handling of unseen data improved the reliability of evaluation results. Exact performance metrics and visualizations are available in the notebook.

Tech Stack:

Programming language used was Python. Libraries included pandas, numpy, scikit-learn, matplotlib, and seaborn. The project was developed and executed in a Jupyter Notebook environment.
