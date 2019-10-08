"""We will check with different data transformation techniques here"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)

# plot the training and test sets
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1], label="Training set", s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], label="Test set", s=60)
axes[0].legend(loc='upper left')
axes[0].set_title("Original Data")

# ---------------------------------------------------------------------------------------
"""1. MinMaxScaler: features are simply shifted and scaled between 0 and 1"""

# Preprocessing using MinMaxScaler
# transform the data first
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("transformed shape: {}".format(X_train_scaled.shape))
print("per-feature minimum before scaling: {}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling: {}".format(X_train.max(axis=0)))
print("per-feature minimum after scaling: {}".format(X_train_scaled.min(axis=0)))
print("per-feature maximum after scaling: {}".format(X_train_scaled.max(axis=0)))

axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], label="Training set", s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], label="Test set", s=60)
axes[1].set_title("MinMaxScaler: Scaled Data")

# Effect of preprocessing on Supervised Learning
svm = SVC(C=100)
svm.fit(X_train, y_train)
print("Test Set Accuracy without MinMaxScaler: {:.2f}".format(svm.score(X_test, y_test)))

svm.fit(X_train_scaled, y_train)
print("Test Set Accuracy with MinMaxScaler: {:.2f}".format(svm.score(X_test_scaled, y_test)))

# ---------------------------------------------------------------------------------------
"""2. StandardScaler: preprocessing using zero mean and unit variance scaling"""

# Preprocessing using StandardScaler
# transform the data first
scaler = StandardScaler()
X_train_std_scaled = scaler.fit_transform(X_train)
X_test_std_scaled = scaler.transform(X_test)
print("transformed shape: {}".format(X_train_std_scaled.shape))
print("per-feature minimum after standard scaling: {}".format(X_train_std_scaled.min(axis=0)))
print("per-feature maximum after standard scaling: {}".format(X_train_std_scaled.max(axis=0)))

axes[2].scatter(X_train_std_scaled[:, 0], X_train_std_scaled[:, 1], label="Training set", s=60)
axes[2].scatter(X_test_std_scaled[:, 0], X_test_std_scaled[:, 1], label="Test set", s=60)
axes[2].set_title("StandardScaler: Scaled Data")

for ax in axes:
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
# plt.show(block=True)

svm.fit(X_train_std_scaled, y_train)
print("Test Set Accuracy with StandardScaler: {:.2f}".format(svm.score(X_test_std_scaled, y_test)))

# ---------------------------------------------------------------------------------------
"""3. PCA: Keep only 2 principal components of the data"""

# Preprocessing using StandardScalar and PCA

scalar = StandardScaler()
scalar.fit(cancer.data)
X_scaled = scalar.transform(cancer.data)

pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca_scaled = pca.transform(X_scaled)
print("Original Shape: {}".format(cancer.data.shape))
print("PCA transformed Shape: {}".format(X_pca_scaled.shape))
print("PCA components:\n{}".format(pca.components_))

# let us plot the transformed shape
plt.figure(figsize=(8, 8))
plt.scatter(X_pca_scaled[:, 0], X_pca_scaled[:, 1], cancer.target)
plt.legend(cancer.target_names, loc="best")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
# plt.show(block=True)

