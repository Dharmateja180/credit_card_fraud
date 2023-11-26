# credit_card_fraud

1. **Import Libraries:**
   ```python
   import pandas as pd
   import numpy as np
   from sklearn import preprocessing
   from sklearn.metrics import confusion_matrix, accuracy_score
   from sklearn import svm
   from sklearn.ensemble import RandomForestClassifier
   import itertools
   import matplotlib.pyplot as plt
   import seaborn
   %matplotlib inline
   ```
   - Import necessary libraries, including pandas for data manipulation, numpy for numerical operations, scikit-learn for machine learning tasks, and matplotlib/seaborn for visualization.

2. **Read Data:**
   ```python
   data = pd.read_csv('creditcard.csv')
   df = pd.DataFrame(data)
   ```
   - Read the dataset from a CSV file and convert it into a Pandas DataFrame (`df`).

3. **Data Exploration and Visualization:**
   ```python
   df.describe()
   df_fraud = df[df['Class'] == 1]
   plt.figure(figsize=(15, 10))
   plt.scatter(df_fraud['Time'], df_fraud['Amount'])
   plt.title('Scatter plot amount fraud')
   plt.xlabel('Time')
   plt.ylabel('Amount')
   plt.xlim([0, 175000])
   plt.ylim([0, 2500])
   plt.show()
   ```
   - Display summary statistics of the dataset using `describe`.
   - Create a scatter plot of fraud amounts over time to visualize patterns.

4. **Data Imbalance Analysis:**
   ```python
   nb_big_fraud = df_fraud[df_fraud['Amount'] > 1000].shape[0]
   print('There are only ' + str(nb_big_fraud) + ' frauds where the amount was bigger than 1000 over ' + str(df_fraud.shape[0]) + ' frauds')

   number_fraud = len(data[data.Class == 1])
   number_no_fraud = len(data[data.Class == 0])
   print('There are only ' + str(number_fraud) + ' frauds in the original dataset, even though there are ' + str(number_no_fraud) + ' no frauds in the dataset.')
   ```
   - Check for frauds with amounts larger than 1000.
   - Check the number of frauds in the dataset.

5. **SVM Model Training:**
   ```python
   df_train_all = df[0:150000]
   df_train_1 = df_train_all[df_train_all['Class'] == 1]
   df_train_0 = df_train_all[df_train_all['Class'] == 0]
   
   print('In this dataset, we have ' + str(len(df_train_1)) + " frauds so we need to take a similar number of non-fraud")
   df_sample = df_train_0.sample(300)
   df_train = df_train_1.append(df_sample)
   df_train = df_train.sample(frac=1)
   
   X_train = df_train.drop(['Time', 'Class'], axis=1)
   y_train = df_train['Class']
   X_train = np.asarray(X_train)
   y_train = np.asarray(y_train)
   ```
   - Prepare the training data by sampling non-fraud instances to balance the dataset.
   - Create feature and label arrays for the SVM model.

6. **SVM Model Training and Evaluation:**
   ```python
   classifier_svm = svm.SVC(kernel='linear')
   classifier_svm.fit(X_train, y_train)
   prediction_svm_all = classifier_svm.predict(X_test_all)
   
   # Evaluate SVM
   accuracy_svm_all = accuracy_score(y_test_all, prediction_svm_all)
   print("Accuracy (SVM - Entire Test Dataset):", accuracy_svm_all)
   ```
   - Create an SVM classifier with a linear kernel and train it on the training data.
   - Make predictions on the entire test dataset.
   - Evaluate the SVM model using accuracy.

7. **Random Forest Model Training and Evaluation:**
   ```python
   classifier_rf = RandomForestClassifier(n_estimators=100, random_state=42)
   classifier_rf.fit(X_train, y_train)
   prediction_rf_all = classifier_rf.predict(X_test_all)
   
   # Evaluate Random Forest
   accuracy_rf_all = accuracy_score(y_test_all, prediction_rf_all)
   print("Accuracy (Random Forest - Entire Test Dataset):", accuracy_rf_all)
   ```
   - Create a Random Forest classifier with 100 trees and a fixed random state.
   - Train the Random Forest model on the training data.
   - Make predictions on the entire test dataset.
   - Evaluate the Random Forest model using accuracy.

we'll use `AdaBoostClassifier` from scikit-learn to combine the SVM and Random Forest models.

```python
from sklearn.ensemble import AdaBoostClassifier

# Create weak learners (base models)
svm_base_model = svm.SVC(kernel='linear', probability=True)
rf_base_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create AdaBoost ensemble
ensemble_model = AdaBoostClassifier(
    base_estimator=None,  # We use default base models (DecisionTreeClassifier)
    n_estimators=50,       # Number of weak learners (you can adjust this)
    learning_rate=1.0,
    random_state=42
)

# Train AdaBoost ensemble on the training data
ensemble_model.fit(X_train, y_train)

# Make predictions on the entire test dataset using the ensemble
prediction_ensemble_all = ensemble_model.predict(X_test_all)

# Evaluate the ensemble model using accuracy
accuracy_ensemble_all = accuracy_score(y_test_all, prediction_ensemble_all)
print("Accuracy (Ensemble - Entire Test Dataset):", accuracy_ensemble_all)
```


8. **Create Weak Learners:**
   - `svm_base_model`: An SVM classifier with a linear kernel and probability estimates enabled.
   - `rf_base_model`: A Random Forest classifier.

9. **Create AdaBoost Ensemble:**
   - Use `AdaBoostClassifier` with the default base model (DecisionTreeClassifier).
   - `n_estimators`: The number of weak learners in the ensemble (you can adjust this based on experimentation).
   - `learning_rate`: The contribution of each weak learner to the final prediction.

10. **Train AdaBoost Ensemble:**
   - Fit the AdaBoost ensemble on the training data (`X_train`, `y_train`).

11. **Make Predictions and Evaluate:**
   - Use the ensemble model to make predictions on the entire test dataset (`X_test_all`).
   - Evaluate the ensemble model's accuracy using `accuracy_score`.

![image](https://github.com/Dharmateja180/credit_card_fraud/assets/106651499/0462ed59-d2f1-4725-8903-c80fcd6d0d58)
