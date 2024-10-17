# Ex.No: 7 Implementation of Decision Tree Learning 
## DATE: 19/09/2024
## REGISTER NUMBER : 212221240041
# AIM:
Design a decision tree for following data. 
1 Healthy, In Cover, With Ammo -> Attack
2.Hurt, In Cover, With Ammo -> Attack
3.Healthy, In Cover, Empty -> Defend
4.Hurt, In Cover, Empty -> Defend
5.Hurt, Exposed, With Ammo -> Defend
# Algorithm:
1. Start the program
2. import the necessary packages 
3. Design a training data and test data 
4. Create a decision tree classifier model
5. Output the predictions 
6. Visualize the decision tree 
# Program:
```
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

# Data: [Health, Cover, Ammo, Exposed]
# Health: 1 for Healthy, 0 for Hurt
# Cover: 1 for In Cover, 0 for Exposed
# Ammo: 1 for With Ammo, 0 for Empty
# Exposed: 1 for Exposed, 0 for In Cover
# Actions: 0 for Defend, 1 for Attack

# Define the training data and corresponding labels (actions)
data = [
    [1, 1, 1],  # Healthy, In Cover, With Ammo -> Attack
    [0, 1, 1],  # Hurt, In Cover, With Ammo -> Attack
    [1, 1, 0],  # Healthy, In Cover, Empty -> Defend
    [0, 1, 0],  # Hurt, In Cover, Empty -> Defend
    [0, 0, 1],  # Hurt, Exposed, With Ammo -> Defend
]

# Labels: 1 for Attack, 0 for Defend
labels = [1, 1, 0, 0, 0]

# Create a Decision Tree Classifier model
clf = DecisionTreeClassifier(criterion="entropy")

# Train the model
clf.fit(data, labels)

# Make predictions (example data points)
test_data = [
    [1, 1, 1],  # Healthy, In Cover, With Ammo
    [0, 0, 1],  # Hurt, Exposed, With Ammo
    [1, 1, 0],  # Healthy, In Cover, Empty
]

# Predict actions for the test data
predictions = clf.predict(test_data)

# Output the predictions
for i, pred in enumerate(predictions):
    action = "Attack" if pred == 1 else "Defend"
    print(f"Test case {i+1}: Predicted action is {action}")

# Optional: Visualize the decision tree
plt.figure(figsize=(10, 6))
tree.plot_tree(clf, feature_names=['Health', 'Cover', 'Ammo'], class_names=['Defend', 'Attack'], filled=True)
plt.show()
```





# Output:

![ex7](https://github.com/user-attachments/assets/cbb549bc-2f06-4ab3-81a8-bed92829dbd0)


# Result:
Thus the optimum value of max player was found using minimax search.
