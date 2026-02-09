# Capastone Project : CyberThreat Detection

This notebook details a comprehensive analysis of  of network intrusion detection using the `cyberfeddefender_dataset.csv` dataset, focusing on building a classification model to detect network intrusions. The process covers data preparation, exploratory data analysis, feature engineering, and baseline model development and evaluation.

## Assignment Notebook

[Click here to view the assignment notebook](https://github.com/ojbskvasu/Capastone_CyberThreat_Detection/blob/main/cyberthreat_detection.ipynb)

### 1. Data Loading
- The `cyberfeddefender_dataset.csv` file was successfully loaded into a pandas DataFrame. This initial step involved verifying the successful import and a quick glance at the dataset's structure.

### 2. Data Cleaning
- **Missing Values**: A thorough check for missing values across all columns revealed that the dataset was remarkably clean, with no `null` entries. This negated the need for any imputation strategies.
- **Duplicate Rows**: Similarly, an inspection for duplicate rows found no exact duplicates, indicating high data integrity and eliminating the need for duplicate removal.

### 3. Exploratory Data Analysis (EDA)
- **Descriptive Statistics**: Detailed descriptive statistics were generated for both numerical and categorical columns. This provided insights into the central tendency, dispersion, and distribution of numerical features, and frequency counts for categorical ones.
- **Data Types**: Initial data type analysis confirmed several `object` type columns, notably `Timestamp`, `Source_IP`, `Destination_IP`, `Protocol`, `Flags`, and `Attack_Type`, alongside numerical `int64` and `float64` types.
- **Distribution Plots**: Histograms with Kernel Density Estimates (KDE) were created for key numerical features (`Packet_Length`, `Duration`, `Bytes_Sent`, `Bytes_Received`, `Flow_Packets/s`) to visualize their distributions. Count plots were used for categorical features (`Protocol`, `Attack_Type`, `Label`) to show their frequency.

### 4. Outlier Analysis
- **Box Plots**: Box plots were generated for all numerical features. These visualizations revealed the presence of potential outliers in several columns, indicating values that fall outside the typical data range. While outliers were identified, no specific removal or imputation strategy was applied at this baseline stage, but it is noted as a point for future investigation.

### 5. Feature Engineering
- **Timestamp Conversion**: The `Timestamp` column, initially an `object` type, was converted to a `datetime` object, enabling time-series based feature extraction.
- **Time-based Features**: New features such as `Hour`, `DayOfWeek`, `Month`, and `Year` were extracted from the `Timestamp` column to capture temporal patterns.
- **Total Bytes Transferred**: A new feature, `Total_Bytes_Transferred`, was created by summing `Bytes_Sent` and `Bytes_Received` to represent the total data volume per flow.
- **Bytes Ratio**: Another feature, `Bytes_Ratio`, was engineered by dividing `Bytes_Sent` by `Bytes_Received` (with a small offset to prevent division by zero). This ratio can help identify asymmetric data flows.

### 6. Classification Modeling Process
- **Data Separation**: Features (X) and the target variable (y, 'Label') were separated. The original `Timestamp` column was dropped from features as its components were already extracted.
- **Categorical Encoding**: Categorical features (`Source_IP`, `Destination_IP`, `Protocol`, `Flags`, `Attack_Type`) were identified and transformed using one-hot encoding via `pd.get_dummies()`, expanding the feature space significantly.
- **Data Splitting**: The dataset was split into training (80%) and testing (20%) sets using `train_test_split`, with `stratify=y` to maintain the original class distribution in both sets.
- **Feature Scaling**: Numerical features were scaled using `StandardScaler` on both training and test sets to address convergence issues encountered with the Logistic Regression model, ensuring that all features contribute equally to the distance calculations.

### 7. Evaluation Metric Selection and Rationale
- Given the context of network intrusion detection, where **false negatives (missed attacks) are more critical than false positives (false alarms)**, a suite of metrics was chosen:
    - **Recall**: Prioritized for its ability to minimize missed attacks.
    - **Precision**: Important for managing false positives and preventing alert fatigue.
    - **F1-Score**: Provides a balanced measure between precision and recall.
    - **ROC AUC**: Chosen for its robustness to class imbalance and its overall assessment of the model's discriminative power across various thresholds.
- Accuracy was deemed less suitable due to potential misleading results in scenarios with class imbalance.

### 8. Baseline Model Performance (Logistic Regression)
- **Model Training**: A Logistic Regression model was chosen as the baseline and trained on the scaled training data.
- **Performance Evaluation**: The model's performance on the test set was evaluated using the selected metrics:
    - **Accuracy**: 0.47
    - **Precision, Recall, F1-score**: Ranged from 0.46 to 0.48 for both classes.
    - **ROC AUC Score**: 0.47
- **Visualization**: A Confusion Matrix and an ROC Curve were generated and saved to the `/images` folder, visually confirming the model's performance.
## How to Run

1. Clone this repository:

   ```bash
   https://github.com/ojbskvasu/Capastone_CyberThreat_Detection.git
   ```
