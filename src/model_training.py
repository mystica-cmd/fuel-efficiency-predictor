import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load cleaned dataset
df = pd.read_csv('data/cleaned_vehicles.csv')

# Prepare input (X) and target (y)
X = df.drop(columns=['comb08'])
y = df['comb08']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

# Store results
results = {}
mae_scores = {}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = (y_test, y_pred)
    mae_scores[name] = mae

    print(f"🔍 {name}")
    print(f"   MAE: {mae:.2f}")
    print(f"   R² Score: {r2:.2f}")
    print("-" * 40)

# ------------------------------------------
# 📈 Plot Actual vs Predicted for each model
# ------------------------------------------
plt.figure(figsize=(18, 5))
for i, (name, (actual, predicted)) in enumerate(results.items()):
    plt.subplot(1, 3, i + 1)
    plt.scatter(actual, predicted, alpha=0.4, label='Predictions')
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    plt.title(f'{name}')
    plt.xlabel('Actual MPG')
    plt.ylabel('Predicted MPG')
    plt.grid(True)

plt.tight_layout()
plt.suptitle("Actual vs Predicted MPG", fontsize=16, y=1.05)
plt.show()

# ------------------------------------------
# 📊 MAE Comparison Bar Chart
# ------------------------------------------
plt.figure(figsize=(6, 4))
plt.bar(mae_scores.keys(), mae_scores.values(), color='skyblue')
plt.title("Model Comparison - Mean Absolute Error (MAE)")
plt.ylabel("MAE")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# ------------------------------------------
# 🔍 Feature Importance (Random Forest)
# ------------------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='mediumseagreen')
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ------------------------------------------
# 💾 Save the Random Forest model
# ------------------------------------------
model_save_path = 'data/random_forest_model.pkl'
joblib.dump(rf, model_save_path)

print(f"✅ Random Forest model saved to {model_save_path}")
