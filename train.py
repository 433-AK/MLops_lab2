import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/credit_card_default.csv')

# Remove ID column (doesn't help prediction)
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split data to avoid overfitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features (important for better performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model (ensure you change this for other branches)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Save metrics
acc = accuracy_score(y_test, preds)
with open('metrics.json', 'w') as f:
    json.dump({'accuracy': acc}, f)

# Generate and save confusion matrix plot
cm = confusion_matrix(y_test, preds, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
