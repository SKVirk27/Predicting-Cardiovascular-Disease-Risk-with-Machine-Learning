import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
# Stacking
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Read csv file
data = pd.read_csv("/content/drive/MyDrive/Imputed_1.csv")

# Preprocess data
target = 'EverHadCVD'
features = data.drop(target, axis=1)
features = features.drop(['SEQN', "EverHadCHD", "EverHadAngina", "EverHadHeartAttack", "EverHadStroke"], axis=1)
target_values = data[target]

numeric_features = features.select_dtypes(include=[np.number])
non_numeric_features = features.select_dtypes(exclude=[np.number])
encoded_features = pd.get_dummies(non_numeric_features)
all_features = pd.concat([numeric_features, encoded_features], axis=1)

# Extra Trees Classifier for feature selection
et_clf = ExtraTreesClassifier(n_estimators=100, random_state=0, n_jobs=-1)
et_clf.fit(all_features, target_values)

importances = et_clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Select the top 25 features
top_n = 25
top_features = all_features.columns[indices[:top_n]]
top_importances = importances[indices[:top_n]]

# Create a DataFrame with the top features and their importances
top_feature_data = pd.DataFrame({"Feature": top_features, "Importance": top_importances})

# Plot the top 25 feature importances
plt.figure(figsize=(10, 5))
sns.set_style("whitegrid")
sns.barplot(data=top_feature_data, x="Feature", y="Importance", palette="viridis")
plt.xticks(rotation=90, fontsize=12)
plt.xlabel("Selected Feature", fontsize=14)
plt.ylabel("Importance", fontsize=14)
plt.title("Top 25 Feature Importances", fontsize=16)
plt.savefig("top_25_feature_importances.png", bbox_inches='tight')
plt.show()

# Load the selected features
selected_data = all_features[top_features]

# Create a StratifiedKFold instance for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Create a Logistic Regression model
model = LogisticRegression(solver='liblinear', random_state=0)

# Perform cross-validation and print the average accuracy
accuracy_scores = cross_val_score(model, selected_data, target_values, cv=cv, scoring='accuracy')
print("Cross-validated accuracy scores:", accuracy_scores)
print("Average cross-validated accuracy:", np.mean(accuracy_scores))

# Determine mutual information between features and target
mutual_info_scores = mutual_info_classif(all_features, target_values)

# Select the top 25 features based on mutual information scores
selector = SelectKBest(mutual_info_classif, k=25)
selector.fit(all_features, target_values)
selected_features_mi = all_features.columns[selector.get_support()]

selected_features_df = all_features[selected_features_mi]

# Create a random forest classifier
rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)

# Calculate cross-validated accuracy scores
cv_scores = cross_val_score(rf_clf, selected_features_df, target_values, cv=5
, scoring='accuracy')

#Print the average cross-validated accuracy
print("Cross-validated accuracy scores:", cv_scores)
print("Average cross-validated accuracy:", np.mean(cv_scores))

#Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(selected_features_df, target_values, test_size=0.3, random_state=0)

#Fit the random forest classifier to the training data
rf_clf.fit(X_train, y_train)

#Predict the test data
y_pred = rf_clf.predict(X_test)

#Print the classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Use the XGBoost model for predictions
xgb_clf = xgb.XGBClassifier(random_state=0, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss')

#Perform a grid search to find the best parameters for the XGBoost model
params = {
'max_depth': [3, 4, 5],
'n_estimators': [50, 100, 150],
'learning_rate': [0.01, 0.1, 0.2]
}
grid_search = GridSearchCV(xgb_clf, param_grid=params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

#Print the best parameters found
print("Best parameters found:", grid_search.best_params_)

#Train the XGBoost model using the best parameters
best_xgb = grid_search.best_estimator_

#Predict the test data
y_pred_xgb = best_xgb.predict(X_test)

#Print the classification report and confusion matrix for the XGBoost model
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# Instantiate the individual classifiers
rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
xgb_clf = xgb.XGBClassifier(random_state=0, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss')
log_clf = LogisticRegression(random_state=0)

# Define the stacking classifier
stacking_clf = StackingClassifier(
    estimators=[('random_forest', rf_clf), ('xgboost', xgb_clf)],
    final_estimator=log_clf
)

# Fit the stacking classifier to the training data
stacking_clf.fit(X_train, y_train)

# Predict the test data
y_pred_stacking = stacking_clf.predict(X_test)

# Print the classification report and confusion matrix for the stacking model
print("Stacking Classifier Classification Report:")
print(classification_report(y_test, y_pred_stacking))

print("Stacking Classifier Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_stacking))