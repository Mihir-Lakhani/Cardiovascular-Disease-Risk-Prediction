import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df=pd.read_csv("cardio_train.csv", sep=";")

# Convert categorical features to numeric manually
def encode_categories(df):
    encodings = {}
    for column in df.columns:
        if df[column].dtype == "object" or df[column].dtype.name == "category":
            unique_values = df[column].unique()
            encodings[column] = {val: idx for idx, val in enumerate(unique_values)}
            df[column] = df[column].map(encodings[column])
    return df, encodings




# Display initial information
print(df.head())
print(df.info())  
print(df.describe())  
print(df.isnull().sum())  
print(df.dtypes)  

# Convert age from days to years
df["age"] = df["age"] // 365

# Fill missing values with the median
df.fillna(df.median(), inplace=True)

# Drop the "id" column (modifies df in place)
df.drop(columns=["id"], inplace=True)

# Categorize age into bins
df["age_category"] = pd.cut(
    df["age"], 
    bins=[19, 29, 39, 49, 59, 69],  
    labels=["20s", "30s", "40s", "50s", "60s"]
)

# Calculate BMI and categorize it
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
df["bmi_category"] = pd.cut(
    df["bmi"], 
    bins=[0, 18.5, 25, 30, 100],  
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)

# Categorize blood pressure (BP)
df["bp_category"] = pd.cut(
    df["ap_hi"]/df["ap_lo"], 
    bins=[0, 119, 139, 159, 200],  
    labels=["Normal", "Elevated", "Hypertension-1", "Hypertension-2"]
)

df["bp"]=df["ap_hi"]/df["ap_lo"]


df['age_bmi_interaction'] = df['age'] * df['bmi']
df['chol_gluc_interaction'] = df['cholesterol'] * df['gluc']
# Create interaction feature: BMI × Smoking
df["bmi_smoke"] = df["bmi"] * df["smoke"]

# Create interaction feature: Age × Physical Activity
df["age_active"] = df["age"] * df["active"]

# Create interaction feature: Blood Pressure × Cholesterol
df["bp_chol"] = df["bp"] * df["cholesterol"]

# Keep only categorical features for ID3
df = df[["age_bmi_interaction", "bp_category", "chol_gluc_interaction", "bmi", "smoke", "age", "active", "bp_chol", "cardio"]]


print(df)

# Drop NaN values created by binning
df.dropna(inplace=True)

# Visualizations--------------------

# Convert categorical columns to numeric codes
df_encoded, encodings = encode_categories(df)


class_counts = df['cardio'].value_counts()

print(class_counts)


# Plot the class distribution
df['cardio'].value_counts().plot(kind='bar', color=['blue', 'red'])
plt.xlabel('Cardiovascular Disease (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()


# Generate and display the correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Implementing ID3 Decision Tree-------------------

# Function to calculate entropy
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-9))  # Small value to prevent log(0)

# Function to calculate information gain
def information_gain(X, y, feature_index):
    unique_values = np.unique(X[:, feature_index])
    weighted_entropy = 0

    for value in unique_values:
        subset_y = y[X[:, feature_index] == value]
        weighted_entropy += (len(subset_y) / len(y)) * entropy(subset_y)

    return entropy(y) - weighted_entropy

# Function to build the ID3 decision tree
def id3(X, y, features):
    if len(set(y)) == 1:  # If all labels are the same
        return y[0]

    if len(features) == 0:  # If no more features to split
        return np.bincount(y).argmax()

    # Select best feature based on information gain
    gains = [information_gain(X, y, i) for i in range(len(features))]
    best_feature = np.argmax(gains)

    tree = {features[best_feature]: {}}
    unique_values = np.unique(X[:, best_feature])

    for value in unique_values:
        subset_indices = X[:, best_feature] == value
        subtree = id3(X[subset_indices], y[subset_indices], np.delete(features, best_feature))
        tree[features[best_feature]][value] = subtree

    return tree


# Convert categorical columns to numeric codes
df_encoded, encodings = encode_categories(df)

# Split into features (X) and target (y)
X = df_encoded.drop(columns=["cardio"]).values  # Features
y = df_encoded["cardio"].values  # Target variable
features = df_encoded.drop(columns=["cardio"]).columns.tolist()  # Feature names list
#--------------------------------------------------------------------
# Build the decision tree for the cardio dataset
#decision_tree = id3(X, y, features)
#--------------------------------------------------------------------
# Replace this:
# decision_tree = id3(X, y, features)

# With training on the 80%:

# Shuffle the DataFrame before splitting
df_shuffled = df_encoded.sample(frac=1).reset_index(drop=True)
        
# 80% training, 20% testing
train_size = int(0.8 * len(df_shuffled))
train_df = df_shuffled[:train_size]
test_df = df_shuffled[train_size:]


features = df_encoded.columns.tolist()
features.remove("cardio")

X_train = train_df[features]
y_train = train_df["cardio"]

X_test = test_df[features]
y_test = test_df["cardio"]

decision_tree = id3(X_train.values, y_train.values, np.array(features))

#decision_tree = id3(train_df, features, "cardio")


# Print the tree
#print(decision_tree)

from graphviz import Digraph

# Function to visualize the ID3 decision tree
def visualize_tree(tree, parent=None, edge_label="Root", graph=None):
    if graph is None:
        graph = Digraph()
        graph.node("Root", label="Root")  # Starting node

    for feature, branches in tree.items():
        for value, subtree in branches.items():
            node_id = f"{feature}_{value}"
            graph.node(node_id, label=f"{feature}={value}")

            if parent is not None:
                graph.edge(parent, node_id, label=edge_label)

            if isinstance(subtree, dict):  # Recursively plot child nodes
                visualize_tree(subtree, parent=node_id, edge_label=str(value), graph=graph)
            else:  # Leaf node
                leaf_id = f"Leaf_{subtree}"
                graph.node(leaf_id, label=f"Class {subtree}", shape="box")
                graph.edge(node_id, leaf_id, label=str(value))

    return graph

# Generate the tree visualization
dot = visualize_tree(decision_tree)
dot.graph_attr['dpi'] = '300'
dot.render("decision_tree", format="pdf", cleanup=False)  # Save as PNG
dot.view()  # View the tree

#TRAINING THE MODEL
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree  # Reached a leaf node

    feature = next(iter(tree))
    value = sample[feature]

    if value in tree[feature]:
        return predict(tree[feature][value], sample)
    else:
        # If the value is not in the tree (unseen during training), return majority class
        return 0


# Reset index to align predictions with true labels
y_test = y_test.reset_index(drop=True)
# Apply prediction to each row
y_pred = X_test.apply(lambda row: predict(decision_tree, row), axis=1)
y_pred = y_pred.values


# Compare with actual
correct = (y_pred == y_test).sum()
total = len(y_test)
accuracy = correct / total




# Reset index to align predictions with true labels
#y_test = y_test.reset_index(drop=True)


# Simulate probabilities: 0.6 for predicted 1, 0.4 for predicted 0
y_scores = np.where(y_pred == 1, 0.6, 0.4)

thresholds = np.linspace(0, 1, 100)
tpr_list = []
fpr_list = []

for thresh in thresholds:
    pred_labels = np.where(y_scores >= thresh, 1, 0)
    
    TP = ((pred_labels == 1) & (y_test == 1)).sum()
    TN = ((pred_labels == 0) & (y_test == 0)).sum()
    FP = ((pred_labels == 1) & (y_test == 0)).sum()
    FN = ((pred_labels == 0) & (y_test == 1)).sum()

    TPR = TP / (TP + FN + 1e-9)
    FPR = FP / (FP + TN + 1e-9)
    
    tpr_list.append(TPR)
    fpr_list.append(FPR)

plt.figure()
plt.plot(fpr_list, tpr_list, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Approximate)')
plt.grid(True)
plt.show()


y_true = np.array(y_test)
y_pred = np.array(y_pred)

# Calculate TP, TN, FP, FN
TP = np.sum((y_pred == 1) & (y_true == 1))
TN = np.sum((y_pred == 0) & (y_true == 0))
FP = np.sum((y_pred == 1) & (y_true == 0))
FN = np.sum((y_pred == 0) & (y_true == 1))

print("📊 Confusion Matrix:")
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")


# Create confusion matrix array
cm = np.array([[TN, FP],
               [FN, TP]])

# Plot the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Preciion Recall Curve
precision_list = []
recall_list = []

for thresh in thresholds:
    pred_labels = np.where(y_scores >= thresh, 1, 0)
    
    TP = ((pred_labels == 1) & (y_test == 1)).sum()
    FP = ((pred_labels == 1) & (y_test == 0)).sum()
    FN = ((pred_labels == 0) & (y_test == 1)).sum()

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)

    precision_list.append(precision)
    recall_list.append(recall)

plt.figure()
plt.plot(recall_list, precision_list, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Approximate)')
plt.grid(True)
plt.show()


# Accuracy Score
correct_predictions = (y_pred == y_test).sum()
total_predictions = len(y_test)
accuracy = correct_predictions / total_predictions
print(f"✅ Accuracy: {accuracy:.5f}")

# Precision, Recall, and F1 Score
tp = cm[1][1]
fp = cm[0][1]
fn = cm[1][0]

# Avoid division by zero
precision_score = tp / (tp + fp) if (tp + fp) != 0 else 0
recall_score = tp / (tp + fn) if (tp + fn) != 0 else 0
f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) != 0 else 0

print(f"Precision: {precision_score:.5f}")
print(f"Recall: {recall_score:.5f}")
print(f"F1 Score: {f1_score:.5f}")

