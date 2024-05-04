# Khai báo thư viện
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier


# Hàm tách các thuộc tính và biến mục tiêu
def splitdataset(balance_data):
    # Biến X chứa các thuộc tính
    X = balance_data.values[:, 1:5]    
    # Biến Y chứa biến mục tiêu
    Y = balance_data.values[:, 0]

    # Chia bộ dữ liệu thành training và testing dataset với tỉ lệ 70:30 (70% cho training, 30% cho test)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test


# Import dataset
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
        sep=',', header=None)

# Chia dataset thành training và test set
X, Y, X_train, X_test, y_train, y_test = splitdataset(data)


# Mô hình Random Forest
clf = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy of Random Forest:", accuracy_score(y_test, y_pred))


# Mô hình Gradient Boosting
clf = GradientBoostingClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy of Gradient Boosting:", accuracy_score(y_test, y_pred))