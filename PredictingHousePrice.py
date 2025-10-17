import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Create the DataFrame from the image data
data = {
    'House': [1, 2, 3, 4, 5],
    'Area': [1200, 1400, 1600, 1700, 1850],
    'Rooms': [3, 4, 3, 5, 4],
    'Distance': [5, 3, 8, 2, 4],
    'Age': [10, 3, 20, 15, 7],
    'Price': [120, 150, 130, 180, 170]
}
df = pd.DataFrame(data).set_index('House')

# 2. Define Features (X) and Target (y)
features = ['Area', 'Rooms', 'Distance', 'Age']
target = 'Price'

X = df[features]
y = df[target]

# 3. Create and Train the Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# 4. Show the Results (Coefficients and Intercept)
print(f"--- House Price Prediction Model ---")

# The intercept (B0)
print(f"Intercept (B0): {model.intercept_:.2f} Lacs")

# The coefficients (B1, B2, B3, B4)
print("\nCoefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"  - {feature}: {coef:.2f}")

print("\n--- Interpretation ---")
print(f"Final Model: Price = {model.intercept_:.2f} + "
      f"({model.coef_[0]:.2f} * Area) + "
      f"({model.coef_[1]:.2f} * Rooms) + "
      f"({model.coef_[2]:.2f} * Distance) + "
      f"({model.coef_[3]:.2f} * Age)")

# Example: How to predict for a new house
new_house = [[1500, 3, 5, 8]]  # 1500 sqft, 3 rooms, 5km, 8 years
prediction = model.predict(new_house)
print(f"\nExample prediction for a 1500sqft, 3-room, 5km, 8-yr-old house: {prediction[0]:.2f} Lacs")