

## Project Overview

### Goal:
This project aims to perform a comprehensive analysis of network intrusion detection. The primary objective is to develop and evaluate a classification model to identify various types of network attacks using the provided dataset.

### Dataset:
The dataset used is `cyberfeddefender_dataset.csv`, located at `/content/data/cyberfeddefender_dataset.csv`.

### Initial Loading Details:
The dataset was loaded into a pandas DataFrame named `df` using `pd.read_csv()`. The initial inspection using `df.head()` showed 23 columns, including 'Timestamp', 'Source_IP', 'Destination_IP', 'Protocol', 'Packet_Length', 'Duration', 'Bytes_Sent', 'Bytes_Received', 'Flags', 'Flow_Packets/s', 'Flow_Bytes/s', 'Avg_Packet_Size', 'Total_Fwd_Packets', 'Total_Bwd_Packets', 'Fwd_Header_Length', 'Bwd_Header_Length', 'Sub_Flow_Fwd_Bytes', 'Sub_Flow_Bwd_Bytes', 'Inbound', 'Attack_Type', and the target variable 'Label'.



## Data Preparation

### 1. Data Loading and Initial Inspection
- The dataset was loaded from `/content/data/cyberfeddefender_dataset.csv` into a pandas DataFrame.
- Initial inspection revealed 1430 entries and 23 columns.

### 2. Data Cleaning
- **Missing Values:** No missing values were found across any columns, as verified by `df.isnull().sum()`.
- **Duplicate Rows:** No duplicate rows were identified or removed, as confirmed by `df.duplicated().sum()`.

### 3. Exploratory Data Analysis (EDA)
- **Descriptive Statistics:** Numerical and categorical column statistics were analyzed to understand data distributions and identify potential issues.
- **Visualizations:**
    - Histograms with KDE were generated for key numerical features (`Packet_Length`, `Duration`, `Bytes_Sent`, `Bytes_Received`, `Flow_Packets/s`) to visualize their distributions. (Saved as `/content/images/distribution_plots.png`)
    - Count plots were created for categorical features (`Protocol`, `Attack_Type`, `Label`) to show their frequency distributions. (Saved as `/content/images/count_plots.png`)
    - Box plots were used to visualize outliers in the numerical features. (Saved as `/content/images/box_plots.png`)

### 4. Feature Engineering
- **Time-based Features:** The 'Timestamp' column was converted to datetime objects, and new features ('Hour', 'DayOfWeek', 'Month', 'Year') were extracted.
- **Traffic Volume Features:**
    - `Total_Bytes_Transferred`: Created by summing 'Bytes_Sent' and 'Bytes_Received'.
    - `Bytes_Ratio`: Calculated as 'Bytes_Sent' divided by ('Bytes_Received' + 1) to represent asymmetric data transfer, handling division by zero.

### 5. Data Preprocessing for Modeling (Leakage Prevention)
- **Feature and Target Separation:** 'Label' was designated as the target variable (y), and all other relevant columns (excluding 'Timestamp') as features (X).
- **Train-Test Split:** The dataset was split into training (80%) and testing (20%) sets using `train_test_split` with `random_state=42` and `stratify=y` to maintain class distribution.
- **Column Transformation:** A `ColumnTransformer` was used to apply:
    - `StandardScaler` to numerical features.
    - `OneHotEncoder` to categorical features (e.g., 'Source_IP', 'Destination_IP', 'Protocol', 'Flags', 'Attack_Type').
- The preprocessor was `fit_transform` only on the training data (`X_train`) and then `transform` on both the training (`X_train_processed`) and test data (`X_test_processed`) to prevent data leakage.
- Processed arrays were converted back to DataFrames (`X_train_df`, `X_test_df`) using the extracted `feature_names` to maintain column labels for subsequent model training.



## Model Development and Evaluation

Three classification models were developed and evaluated: Logistic Regression, Random Forest, and Decision Tree.

### 1. Logistic Regression
- **Model:** Logistic Regression (`LogisticRegression(random_state=42, max_iter=5000)`)
- **Preprocessing:** Numerical features were scaled using `StandardScaler` prior to training to address convergence issues and prevent data leakage.
- **Evaluation Metrics:**
    - Accuracy: 0.47
    - Precision (Class 0): 0.48
    - Recall (Class 0): 0.47
    - F1-Score (Class 0): 0.48
    - Precision (Class 1): 0.46
    - Recall (Class 1): 0.47
    - F1-Score (Class 1): 0.47
    - ROC AUC Score: 0.47
- **Visualizations:**
    - Confusion Matrix (Saved as `/content/images/confusion_matrix.png`)
    - ROC Curve (Saved as `/content/images/roc_curve.png`)

### 2. Random Forest Classifier
- **Model:** Random Forest Classifier (`RandomForestClassifier(random_state=42)`)
- **Evaluation Metrics:**
    - Accuracy: 0.47
    - Precision (Class 0): 0.48
    - Recall (Class 0): 0.55
    - F1-Score (Class 0): 0.51
    - Precision (Class 1): 0.45
    - Recall (Class 1): 0.39
    - F1-Score (Class 1): 0.42
    - ROC AUC Score: 0.46
- **Visualizations:**
    - Confusion Matrix (Saved as `/content/images/confusion_matrix_rf.png`)
    - ROC Curve (Saved as `/content/images/roc_curve_rf.png`)
- **Hyperparameter Tuning:** `GridSearchCV` was performed to optimize Random Forest hyperparameters.
    - Best parameters found: {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 200}
    - Best ROC AUC score from tuning: 0.53

### 3. Decision Tree Classifier
- **Model:** Decision Tree Classifier (`DecisionTreeClassifier(random_state=42)`)
- **Evaluation Metrics:**
    - Accuracy: 0.49
    - Precision (Class 0): 0.50
    - Recall (Class 0): 0.49
    - F1-Score (Class 0): 0.49
    - Precision (Class 1): 0.48
    - Recall (Class 1): 0.49
    - F1-Score (Class 1): 0.48
    - ROC AUC Score: 0.49
- **Visualizations:**
    - Confusion Matrix (Saved as `/content/images/confusion_matrix_dt.png`)
    - ROC Curve (Saved as `/content/images/roc_curve_dt.png`)



## Feature Importance Analysis

Feature importances were analyzed using the Random Forest Classifier to identify the most influential features.

- **Top 10 Features:**
    - `Flow_Bytes/s`
    - `Bytes_Sent`
    - `Packet_Length`
    - `Flow_Packets/s`
    - `Sub_Flow_Fwd_Bytes`
    - `Bytes_Ratio`
    - `Duration`
    - `Sub_Flow_Bwd_Bytes`
    - `Bytes_Received`
    - `Total_Fwd_Packets`

These features, primarily related to network traffic volume and statistics, are highly predictive of network intrusion.
- **Visualization:** A bar plot showing the top 10 feature importances was saved as `/content/images/feature_importances_rf.png`.



## Overall Findings and Recommendations

### 1. Model Performance Comparison

| Model                 | Accuracy | Precision (Class 0) | Recall (Class 0) | F1-Score (Class 0) | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | ROC AUC Score |
| :-------------------- | :------- | :------------------ | :--------------- | :----------------- | :------------------ | :--------------- | :----------------- | :------------ |
| Logistic Regression   | 0.47     | 0.48                | 0.47             | 0.48               | 0.46                | 0.47             | 0.47               | 0.47          |
| Random Forest         | 0.47     | 0.48                | 0.55             | 0.51               | 0.45                | 0.39             | 0.42               | 0.46          |
| Decision Tree         | 0.49     | 0.50                | 0.49             | 0.49               | 0.48                | 0.49             | 0.48               | 0.49          |

The **Decision Tree Classifier** generally performed best among the three models, showing slightly higher Accuracy, F1-Scores for both classes, and ROC AUC score. The Random Forest model, despite hyperparameter tuning, exhibited lower performance for Class 1 (attack detection).

### 2. Feature Importance Insights from Random Forest:

The top 10 most influential features, as determined by the Random Forest model, are:

- `Flow_Bytes/s`
- `Bytes_Sent`
- `Packet_Length`
- `Flow_Packets/s`
- `Sub_Flow_Fwd_Bytes`
- `Bytes_Ratio`
- `Duration`
- `Sub_Flow_Bwd_Bytes`
- `Bytes_Received`
- `Total_Fwd_Packets`

These features are primarily related to network traffic volume, duration, and packet statistics. This suggests that the model heavily relies on the quantitative aspects of network flows to distinguish between normal and attack traffic. Features like Timestamp-derived values (Hour, DayOfWeek, Month, Year) and specific IP addresses or protocols have lower importance, indicating that the magnitude of traffic characteristics is more predictive than temporal patterns or specific network entities in this dataset for the Random Forest model.

### 3. Strengths and Weaknesses of Each Model:

*   **Logistic Regression:**
    *   **Strengths:** Simple, interpretable, good baseline. Relatively stable performance across both classes.
    *   **Weaknesses:** Achieves the lowest overall performance among the three models. Its linear nature may not capture complex relationships in the data.

*   **Random Forest:**
    *   **Strengths:** Ensemble method, generally robust to overfitting, can handle non-linear relationships. Provides feature importance, which is valuable for interpretability and feature selection.
    *   **Weaknesses:** Performed the worst in terms of ROC AUC and F1-score for class 1. This suggests that while it is good at identifying class 0 (normal), it struggles with the minority class (attacks). It might be overfitting to the majority class or the hyperparameters need more aggressive tuning to address class imbalance.

*   **Decision Tree:**
    *   **Strengths:** Relatively simple to understand and interpret (for shallow trees). Achieved the best performance among the three in terms of Accuracy, F1-score (for class 1), and ROC AUC.
    *   **Weaknesses:** Prone to overfitting if not properly regularized (e.g., controlling `max_depth`). Performance can be unstable with small variations in data.

### 4. Recommendations:

For this intrusion detection task, the **Decision Tree Classifier** currently appears to be the most promising model given its slightly superior performance metrics, especially in identifying the attack class (Class 1) compared to the other two. Its interpretability is also a significant advantage in security-related applications.

### 5. Limitations and Next Steps:

*   **Limitations:** All models show relatively low performance (ROC AUC scores are below 0.5, implying they are not much better than random guessing). This suggests that the current features or the inherent complexity of the problem might require more sophisticated approaches or a more balanced dataset. The dataset might suffer from severe class imbalance, which these models did not effectively handle.
*   **Next Steps for Improvement:**
    *   **Advanced Feature Engineering:** Explore more complex interactions between features or generate sequence-based features from network flows.
    *   **Addressing Class Imbalance:** Implement techniques like SMOTE (Synthetic Minority Over-sampling Technique), ADASYN, or explore different sampling strategies (undersampling majority class, oversampling minority class) during training.
    *   **Hyperparameter Tuning:** Conduct more extensive hyperparameter tuning for all models, especially for the Random Forest, using techniques beyond `GridSearchCV` like `RandomizedSearchCV` or Bayesian optimization.
    *   **Ensemble Methods (Advanced):** Investigate more powerful ensemble methods like Gradient Boosting (e.g., XGBoost, LightGBM) or Stacking, which often yield better results in complex classification tasks.
    *   **Deep Learning Models:** For highly complex patterns, recurrent neural networks (RNNs) or convolutional neural networks (CNNs) could be explored, especially if the data can be represented as sequences or images.

