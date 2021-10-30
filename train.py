from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle
import numpy as np
from scipy import stats


def normalize(train, test):
    normalized = stats.zscore(np.concatenate((train, test), axis=0), axis=0)
    return normalized[:train.shape[0], :], normalized[train.shape[0]:, :]


with open('X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)


# clf = SVC(kernel='rbf', tol=1e-3, C=10, gamma=1e-5)
clf = RandomForestClassifier(max_depth=5)
clf.fit(X_train, y_train)
print(f"Training score of classifier: {clf.score(X_train, y_train)}")
print(f"Test score of classifier: {clf.score(X_test, y_test)}")