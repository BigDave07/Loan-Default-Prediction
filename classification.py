import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score ,recall_score, f1_score

df = pd.read_csv(r"c:\Users\dell\Downloads\archive (6)\train_u6lujuX_CVtuZ9i (1).csv")

# print(f"{df.head(10)}")
# print(f"{df.shape}")

# df.describe()
# df.info()


# DATA CLEANING
# Check missing values  
# print(df.isnull().sum())
# Fill missing values with most common value called the mode (Categorical Columns)
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

# Fill Numerical Column with an average value instead /median
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
# Double check and see if there are any missing values
# print(df.isnull().sum())

# FEATURE ENGINEERING
# print(df.dtypes.value_counts()) 

# Use LabelEncoder for Binary Columns
# These have only 2 categories — use simple encoding:
#  What this does:
# 'Male' becomes 1, 'Female' becomes 0
# 'Yes' becomes 1, 'No' becomes 0
# 'Y' becomes 1, 'N' becomes 0
le = LabelEncoder()
binary_col = ["Gender", "Married", "Education", "Self_Employed", "Loan_Status"]

for col in binary_col:
    df[col] = le.fit_transform(df[col])
    
# Convert ‘Dependents’ Column
# '3+' should become just 3, then convert the column to numeric:
df["Dependents"]= df["Dependents"].replace('3+', 3)
df["Dependents"]= df["Dependents"].astype(int)

# Use One-Hot Encoding for Multiclass Columns
# For columns with more than 2 unique values (like 'Property_Area'):
df = pd.get_dummies(df, columns=["Property_Area"], drop_first=True)

# print(df.dtypes)
# Create feature(X) amd target(y)
y = df["Loan_Status"]
X = df.drop(columns=["Loan_Status", "Loan_ID"], errors="ignore")

# print("X shape:", X.shape)
# print("y shape:", y.shape)
# print(X.head())
# print(y.value_counts())

# MODEL TRAINING AND COMPARISON

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

# print("Training set: ", X_train.shape, y_train.shape)
# print("Test set: ", X_test.shape, y_test.shape)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Random Tree
rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)
rand_forest.fit(X_train, y_train)

# Xgboost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train, y_train)

# EVALUATE CLASSIFIERS
# Create an Evaluation Function

def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    
    print(f"\n {name} Evaluation:")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
    print(f"Precision Score: {precision_score(y_test, y_pred)}")
    print(f"Recall Score: {recall_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    

evaluate_model(log_reg, X_test, y_test, "Logistic Regression")
evaluate_model(knn, X_test, y_test, "KNN")
evaluate_model(tree, X_test, y_test, "Decision Tree")
evaluate_model(rand_forest, X_test, y_test, "Random Forest")
evaluate_model(xgb_model,  X_test, y_test, "XGBoost")
