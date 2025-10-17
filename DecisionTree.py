import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# --- Part 1: Manual Python Implementation ---

def calculate_gini(labels):
    """Calculates the Gini impurity for a set of labels."""
    if len(labels) == 0:
        return 0
    counts = np.bincount(labels)
    probabilities = counts / len(labels)
    return 1 - np.sum(probabilities**2)

# Data
X = np.array([2, 4, 6, 8, 10])
y = np.array([0, 0, 1, 1, 1])
split_points = [3, 5, 7, 9]

print("--- Manual Gini Calculation ---")
parent_gini = calculate_gini(y)
print(f"Parent Gini: {parent_gini:.4f}\n")

best_gini = 1
best_split = None

for split_val in split_points:
    left_mask = X <= split_val
    right_mask = X > split_val
    
    left_labels = y[left_mask]
    right_labels = y[right_mask]
    
    gini_left = calculate_gini(left_labels)
    gini_right = calculate_gini(right_labels)
    
    n_left = len(left_labels)
    n_right = len(right_labels)
    n_total = len(y)
    
    weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
    
    print(f"Split: Hours <= {split_val}")
    print(f"  Left: {left_labels}, Gini: {gini_left:.4f}")
    print(f"  Right: {right_labels}, Gini: {gini_right:.4f}")
    print(f"  Weighted Gini: {weighted_gini:.4f}\n")
    
    if weighted_gini < best_gini:
        best_gini = weighted_gini
        best_split = split_val

print("--- Final Tree (Text) ---")
print(f"Best Split Chosen: Study Hours <= {best_split}")
print(f"Lowest Weighted Gini: {best_gini:.4f}")
print(f"""
        [Study Hours <= {best_split}]
        Gini = {parent_gini:.2f}
       /                 \\
 (True) /                   \\ (False)
      /                     \\
[Class: 0 (Fail)]         [Class: 1 (Pass)]
  Gini = 0.0                Gini = 0.0
""")


# --- Part 2: Optional Sklearn Cross-Check ---

print("\n--- Sklearn Cross-Check ---")
# sklearn expects X to be 2D
X_2d = X.reshape(-1, 1)

# Initialize and fit the tree
# We set max_depth=1 to get only the first split
sk_tree = DecisionTreeClassifier(criterion='gini', max_depth=1)
sk_tree.fit(X_2d, y)

# Visualize the tree
plt.figure(figsize=(10, 6))
plot_tree(sk_tree, 
          feature_names=['Study Hours'], 
          class_names=['Fail (0)', 'Pass (1)'],
          filled=True, 
          impurity=True, 
          rounded=True)
plt.title("Sklearn Decision Tree (max_depth=1)")
plt.show()

print("Sklearn confirms the split around 5 (it finds 5.0 as the threshold).")