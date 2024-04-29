from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# 4 models
clf_tree = tree.DecisionTreeClassifier()
clf_knn = KNeighborsClassifier()
clf_rf = RandomForestClassifier()
clf_svm = SVC()

# Training
clf_tree.fit(X,Y)
clf_knn.fit(X,Y)
clf_rf.fit(X,Y)
clf_svm.fit(X,Y)

# Testing on same data
pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) *100
print('Acc for Decision Tree: {}'.format(acc_tree))

pred_knn = clf_knn.predict(X)
acc_knn = accuracy_score(Y, pred_knn) *100
print('Acc for KNN: {}'.format(acc_knn))

pred_rf = clf_rf.predict(X)
acc_rf = accuracy_score(Y, pred_rf) *100
print('Acc for Random Forest: {}'.format(acc_rf))

pred_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y, pred_svm) *100
print('Acc for SVM: {}'.format(acc_svm))

