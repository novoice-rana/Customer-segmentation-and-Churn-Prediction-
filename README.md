# 📡 Telco Customer Churn Prediction

A complete end-to-end machine learning project for predicting customer churn using the **Telco Customer Churn dataset**. The project covers exploratory data analysis, data preprocessing, model building with Random Forest, cross-validation, ROC-AUC evaluation, and customer segmentation using K-Means clustering.

---

## 📁 Dataset

- **File:** `Telco_customer_churn.xlsx`
- **Target Variable:** `Churn Value` (1 = Churned, 0 = Retained)
- **Key Features:** Tenure Months, Monthly Charges, Total Charges, Contract Type, Internet Service, Payment Method, Tech Support, and more.

---

## 🔧 Libraries Used

```python
pandas, numpy, matplotlib, seaborn
sklearn (RandomForestClassifier, KMeans, StandardScaler, train_test_split, metrics)
```

---

## 📊 Step 1: Exploratory Data Analysis (EDA)

The EDA phase explores the distribution and relationships of key features with respect to churn.

- **Churn Distribution** — Checked value counts of `Churn Label` (Yes/No) to understand class imbalance.
- **Tenure Months** — Histplot with KDE to see customer tenure distribution; Boxplot comparing churned vs retained customers.
- **Monthly Charges** — Histplot and Boxplot to compare spending patterns between churned and non-churned customers. Quartile analysis was done separately for `Churn = Yes` and `Churn = No`.
- **Contract Type** — Countplot with hue `Churn Label` to identify which contract type has higher churn.
- **Internet Service** — Countplot to see how fiber optic vs DSL vs no service relates to churn.
- **Payment Method** — Countplot showing churn rate across different payment methods.
- **Tech Support** — Countplot to explore whether lack of tech support correlates with churn.
- **Average Tenure by Churn** — `groupby` on `Churn Label` to get mean tenure for churned vs retained.
- **Correlation Matrix** — Heatmap across numerical columns: `Tenure Months`, `Monthly Charges`, `Churn Value`, `Churn Score`, `CLTV`.
- **Cross-Tabulation** — Contract type vs Churn Label (normalized by index) to see churn rates per contract.

---

## 🧹 Step 2: Data Cleaning

- **`Total Charges` conversion** — Column was in object format; converted to numeric using `pd.to_numeric(errors='coerce')`.
- **Missing values** — Null values in `Total Charges` were found to correspond to customers with 0 tenure months; filled with `0`.
- **Dropping irrelevant columns** — The following columns were removed as they are identifiers, redundant, or leak target information:

```python
drop_columns = [
    'CustomerID', 'Count', 'Country', 'State', 'Zip Code',
    'Lat Long', 'Latitude', 'City', 'Longitude',
    'Churn Label', 'Churn Score', 'CLTV', 'Churn Reason'
]
```

---

## 🔢 Step 3: Encoding

- **One-Hot Encoding** applied to all categorical columns using `pd.get_dummies(df, drop_first=True)`.
- This converts string categories (e.g., Contract type, Internet Service) into binary numeric columns.
- Result: The dataset expanded to a larger feature set of encoded columns.

---

## 🎯 Step 4: Feature Selection

- **X (Features):** All columns except `Churn Value`.
- **Y (Target):** `Churn Value` (binary: 0 or 1).
- After training the tuned Random Forest model, **feature importance scores** were extracted and ranked.
- The **bottom 15 least important features** were identified, and the following low-importance features were dropped to create a leaner model:

```python
dropping = [
    'Phone Service_Yes',
    'Multiple Lines_No phone service',
    'Streaming TV_Yes',
    'Streaming Movies_Yes',
    'Device Protection_No internet service'
]
```

- A new model `rf_selected` was trained on `X_selected` (after dropping these features).

---

## ✂️ Step 5: Train and Test Split

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42
)
```

- **80% training / 20% testing** split with a fixed `random_state=42` for reproducibility.
- Same split was applied to `X_selected` for the feature-selected model.

---

## 🌲 Step 6: Random Forest — Three Approaches

### Baseline Model

```python
rf_modal = RandomForestClassifier(n_estimators=100, random_state=42)
```

Evaluated using `accuracy_score`, `confusion_matrix`, and `classification_report`.

---

### Approach i — Handling Class Imbalance

```python
rf_balanced = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced'
)
```

The dataset has more retained customers than churned ones. Using `class_weight='balanced'` tells the model to give more importance to the minority class (churned customers), improving recall for churn detection.

---

### Approach ii — Hyperparameter Tuning

```python
rf_tunned = RandomForestClassifier(
    n_estimators=300, max_depth=10, random_state=42, class_weight='balanced'
)
```

A manual grid search was also performed across combinations of `n_estimators` and `max_depth`:

```python
n_estimators_list = [100, 200, 300, 400, 500]
max_depth_list    = [5, 10, 15, 20, 25]
```

Results were sorted by **Recall** (priority) then **Accuracy** to find the best model for catching churners.

---

### Approach iii — Feature Importance Analysis

```python
feature_importance = pd.DataFrame({
    'features'  : X.columns,
    'importance': rf_tunned.feature_importances_
}).sort_values(by='importance', ascending=False)
```

- Top features identified and printed.
- Bottom 15 features reviewed; 5 low-importance features dropped.
- A new model trained on the reduced feature set to verify performance is maintained.

---

## 🔁 Step 7: Cross-Validation

Cross-validation was implemented (commented out in the notebook, available for use):

```python
from sklearn.model_selection import cross_val_score

final_rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, random_state=42, class_weight='balanced'
)

cv_accuracy = cross_val_score(final_rf, X, Y, cv=5, scoring='accuracy')
cv_recall   = cross_val_score(final_rf, X, Y, cv=5, scoring='recall')
```

- **5-Fold Cross-Validation** used to get a more reliable estimate of model performance.
- Both accuracy and recall scores were computed and averaged across folds.

---

## 📈 Step 8: ROC – AUC Curve

```python
from sklearn.metrics import roc_auc_score, roc_curve

y_prob1    = rf_tunned.predict_proba(X_test)
churn_prob = y_prob1[:, 1]
fpr, tpr, threshold = roc_curve(Y_test, churn_prob)
```

- **`predict_proba`** used to get churn probability scores instead of binary predictions.
- **ROC Curve** plotted using False Positive Rate (FPR) vs True Positive Rate (TPR).
- **AUC Score** computed to measure overall model discrimination ability (higher = better).

---

## 🗂️ Step 9: Customer Segmentation using K-Means (with Churn Prediction)

### Segmentation Features

Churn probability from the tuned Random Forest was combined with financial features:

```python
segmentation_data = pd.DataFrame({
    'Tenure Months'     : X['Tenure Months'],
    'Monthly Charges'   : X['Monthly Charges'],
    'Total Charges'     : X['Total Charges'],
    'Churn Probability' : churn_probability
})
```

### Scaling

```python
scaler      = StandardScaler()
scaled_data = scaler.fit_transform(segmentation_data)
```

### Finding Optimal K — Elbow Method

```python
for k in range(1, 16):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
```

A WCSS (Within-Cluster Sum of Squares) vs K plot was drawn to identify the elbow point — **K=3** was selected.

### Cluster Labels

| Cluster | Segment Name           |
|---------|------------------------|
| 0       | Budget Loyal Customer  |
| 1       | High Risk Customer     |
| 2       | Loyal Premium Customer |

### Cluster Analysis

- `groupby('Cluster').mean()` used to get average feature values per cluster.
- Scatter plots created: **Tenure Months vs Churn Probability** and **Monthly Charges vs Churn Probability**, colored by cluster segment.

---

## 📉 Step 10: Data Visualisation

All visualisations were created using **Matplotlib** and **Seaborn**:

| Chart | Purpose |
|-------|---------|
| Histplot — Tenure Months | Distribution of customer tenure |
| Boxplot — Tenure vs Churn | Tenure comparison for churned/retained |
| Histplot — Monthly Charges | Distribution of monthly billing |
| Boxplot — Monthly Charges vs Churn | Charges comparison for churned/retained |
| Countplot — Contract vs Churn | Contract type and churn correlation |
| Countplot — Internet Service vs Churn | Service type and churn correlation |
| Countplot — Payment Method vs Churn | Payment method and churn correlation |
| Countplot — Tech Support vs Churn | Tech support impact on churn |
| Elbow Curve (WCSS vs K) | Optimal cluster selection for K-Means |
| Scatter — Tenure vs Churn Probability | Cluster visualization |
| Scatter — Monthly Charges vs Churn Probability | Cluster visualization |

---

## 🚀 How to Run

1. Clone or download this repository.
2. Place `Telco_customer_churn.xlsx` in the same directory as the notebook.
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
   ```
4. Open `CBSOTPROJ1.ipynb` in Jupyter Notebook or Google Colab.
5. Run all cells sequentially from top to bottom.

---

## 📌 Key Takeaways

- Customers on **month-to-month contracts** with **higher monthly charges** and **shorter tenure** are the most likely to churn.
- Using `class_weight='balanced'` significantly improves **recall for churned customers**.
- The best performing configuration found via grid search was **300 trees with max depth 10**.
- K-Means clustering with churn probability as a feature identifies three distinct customer risk segments for targeted retention strategies.
