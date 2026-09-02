# ==========================================
# Telco Customer Churn Analysis
# Using PyCharm
# ==========================================

import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 60)
print("Telecom Customer Churn Analysis")
print("=" * 60)

# ==========================================
# Step 1: Load Data
# ==========================================
print("\n[Step 1] Loading data...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"✅ Data loaded! {len(df)} rows, {len(df.columns)} columns")

# ==========================================
# Step 2: Data Cleaning
# ==========================================
print("\n[Step 2] Data cleaning...")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

print(f"✅ Cleaning done! {len(df)} rows remaining")
print(f"📊 Churned customers: {df['Churn'].sum()}")
print(f"📊 Churn rate: {df['Churn'].mean() * 100:.2f}%")

# ==========================================
# Step 3: Descriptive Statistics
# ==========================================
print("\n[Step 3] Descriptive statistics...")
print("\nNumerical columns summary:")
print(df.describe())

# ==========================================
# Step 4: Visualization (6 charts)
# ==========================================
print("\n[Step 4] Generating visualizations...")

sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

# Chart 1: Overall Churn Rate (Pie Chart)
plt.subplot(2, 3, 1)
churn_counts = df['Churn'].value_counts()
plt.pie(churn_counts, labels=['Not Churned', 'Churned'], autopct='%1.1f%%',
        colors=['#66b3ff', '#ff6b6b'], startangle=90, explode=(0, 0.05))
plt.title('1. Overall Churn Rate', fontsize=12, fontweight='bold')

# Chart 2: Contract Type vs Churn
plt.subplot(2, 3, 2)
sns.countplot(data=df, x='Contract', hue='Churn', palette='Set2')
plt.title('2. Churn by Contract Type', fontsize=12, fontweight='bold')
plt.xlabel('Contract Type')
plt.ylabel('Customer Count')
plt.legend(['Not Churned', 'Churned'])

# Chart 3: Online Security vs Churn
plt.subplot(2, 3, 3)
sns.countplot(data=df, x='OnlineSecurity', hue='Churn', palette='Set2')
plt.title('3. Churn by Online Security', fontsize=12, fontweight='bold')
plt.xlabel('Online Security')
plt.ylabel('Customer Count')
plt.legend(['Not Churned', 'Churned'])

# Chart 4: Monthly Charges Comparison
plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette=['#66b3ff', '#ff6b6b'])
plt.title('4. Monthly Charges Comparison', fontsize=12, fontweight='bold')
plt.xticks([0, 1], ['Not Churned', 'Churned'])
plt.ylabel('Monthly Charges (USD)')

# Chart 5: Tenure Comparison
plt.subplot(2, 3, 5)
sns.boxplot(data=df, x='Churn', y='tenure', palette=['#66b3ff', '#ff6b6b'])
plt.title('5. Tenure Comparison', fontsize=12, fontweight='bold')
plt.xticks([0, 1], ['Not Churned', 'Churned'])
plt.ylabel('Tenure (Months)')

# Chart 6: Partner vs Churn
plt.subplot(2, 3, 6)
sns.countplot(data=df, x='Partner', hue='Churn', palette='Set2')
plt.title('6. Churn by Partner Status', fontsize=12, fontweight='bold')
plt.xlabel('Has Partner')
plt.ylabel('Customer Count')
plt.legend(['Not Churned', 'Churned'])

plt.tight_layout()
plt.show()
print("✅ Charts displayed successfully")

# ==========================================
# Step 5: Machine Learning Prediction
# ==========================================
print("\n[Step 5] Machine Learning prediction...")

df_encoded = df.copy()
le = LabelEncoder()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop(['customerID', 'Churn'], axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("🔄 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# Step 6: Feature Importance Analysis
# ==========================================
print("\n[Step 6] Feature importance analysis...")

importances = model.feature_importances_
feature_names = X.columns
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df = feature_df.sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
for i, row in feature_df.head(10).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# Feature Importance Bar Chart
plt.figure(figsize=(10, 8))
top_features = feature_df.head(10)
sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis')
plt.title('Top 10 Features for Churn Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("✅ Analysis Complete!")
print("=" * 60)

print("\n📊 Key Findings:")
print(f"1. Overall churn rate: {df['Churn'].mean() * 100:.2f}%")
print("2. ⚠️ Month-to-month contract customers have much higher churn risk")
print("3. ⚠️ Customers without Online Security are more likely to churn")
print("4. Churned customers have higher monthly charges and shorter tenure")
print(f"5. Most important churn predictor: {feature_df.head(1)['Feature'].values[0]}")