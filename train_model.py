import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Sample data
data = pd.read_csv("transactions.csv")  # Use 100–1000 fake/simulated rows

# Feature engineering (simple)
data['txn_hour'] = pd.to_datetime(data['timestamp']).dt.hour
X = data[['amount', 'txn_hour']]
y = data['is_fraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "fraud_model.pkl")
