import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

train = "IA3-train.csv"
dev = "IA3-dev.csv"

## Part 0: Preprocessing 
def load_data(path, sentiment=None):
    df = pd.read_csv(path)
    y = df['sentiment']
    if sentiment != None:
        X = df[df['sentiment'] == sentiment]
    else:
        X = df
    return X["text"], y

#### Part 0(a) - Bag of words feature extraction
def extract_most_freq_words(bog, vect_inst, n):
    sum_cols = bog.sum(axis=0)
    words_freq = [(word, sum_cols[0, pos]) for word, pos in vect_inst.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    most_freq = words_freq[:n]
    return most_freq

###### positive sentiment
X_pos, y = load_data(train, sentiment=1)
coun_vect = CountVectorizer(lowercase=True)
bag_of_words_pos = coun_vect.fit_transform(X_pos)

print("The Top ten most frequent positive words using CountVectorizer are-->")
for item in extract_most_freq_words(bag_of_words_pos, coun_vect, 10):
    print(item[0], ":", item[1])

###### negative sentiment
X_neg, y = load_data(train, sentiment=0)
coun_vect = CountVectorizer(lowercase=True)
bag_of_words_neg = coun_vect.fit_transform(X_neg)

print("The Top ten most frequent negative words using CountVectorizer are-->")
for item in extract_most_freq_words(bag_of_words_neg, coun_vect, 10):
    print(item[0], ":", item[1])

#### Part 0(b) - TF-IDF feature extraction
def extract_mfw_tfidf(tfidf_mat, vect_inst, n):
    sum_cols = tfidf_mat.sum(axis=0)
    words_freq = [(word, sum_cols[0, pos]) for word, pos in vect_inst.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    most_freq = words_freq[:n]
    return most_freq

print("\n")

###### positive sentiment
X_pos, y = load_data(train, sentiment=1)
tfidf_vect = TfidfVectorizer(use_idf=True, lowercase=True)
tfidf_mat_pos = tfidf_vect.fit_transform(X_pos)

print("The Top ten most frequent positive words using TFIDF vectorizer-->")
for item in extract_mfw_tfidf(tfidf_mat_pos, tfidf_vect, 10):
    print(item[0], ":", item[1])

###### negative sentiment
X_neg, y = load_data(train, sentiment=0)
tfidf_vect = TfidfVectorizer(use_idf=True, lowercase=True)
tfidf_mat_neg = tfidf_vect.fit_transform(X_neg)

print("The Top ten most frequent negative words using TFIDF vectorizer-->")
for item in extract_mfw_tfidf(tfidf_mat_neg, tfidf_vect, 10):
    print(item[0], ":", item[1])

print("\n")

## Part 1: Linear SVM 
def linear_svm(c, bog, label):
    clf = svm.SVC(kernel='linear', C=c)
    clf.fit(bog, label)
    bog_array = bog.toarray()
    y_pred = clf.predict(bog_array)
    acc = accuracy_score(y_pred, label)
    print(f"Accuracy on the linear-svm train set with C value {c}: {acc:.4f}", "\t")
    return clf, acc

###### use the TF-IDF vectorizer on the train data to build your vocabulary
X_train, y_train = load_data(train)
tfidf_vect = TfidfVectorizer(use_idf=True, lowercase=True)
tfidf_mat_train = tfidf_vect.fit_transform(X_train)

###### get the text (X) and labels (y) of the validation data
X_dev, y_dev = load_data(dev)

###### convert the validation data into TF-IDF representations using the vocabulary (vectorizer) built with the trained data
trained_tfidf_vectorizer = tfidf_vect
X_test = trained_tfidf_vectorizer.transform(X_dev)

###### predict using the trained linear classifier and tune the values of the hyperparameter c
c = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
training_acc = []
validation_acc = []
n_support_vectors = []
for i in c:
    linear_clf, train_acc = linear_svm(pow(10, i), tfidf_mat_train, y_train)
    linear_ypred = linear_clf.predict(X_test)
    linear_acc = accuracy_score(linear_ypred, y_dev)
    training_acc.append(train_acc)
    validation_acc.append(linear_acc)
    n_support_vectors.append(sum(linear_clf.n_support_))
    print(f"Accuracy on the linear-svm test set with C value 10^{i}: {linear_acc:.4f}")
    print("---------------------------------------------")

print("\n")

###### plot of the training and validation accuracy for linear svm
print('Plot of training versus validation accuracy...\t')

c = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
fig, ax3 = plt.subplots(figsize=(8, 6), tight_layout=True)
ax3.semilogx(c, training_acc, color='r', marker='o', markerfacecolor='m')
ax3.semilogx(c, validation_acc, color='b', marker='x', markerfacecolor='r')

min_axis = min(min(training_acc), min(validation_acc))
max_axis = max(max(training_acc), max(validation_acc))

ax3.set_ylabel(f'accuracy', color='r')
ax3.set_xlabel(f'c')
ax3.set_xlim([1e-4, 1e5])
ax3.set_ylim(0.7, 1.1)
ax3.set_title(f"Classification Accuracy", color='k', weight='normal', size=10)
ax3.legend(["training", "validation"], loc="upper left")
plt.savefig("../figures/linear_train_dev_acc_cmp.jpg")
print('Done.\n')

print("\n")

###### plot of the number of support vectors versus the regularization parameters, c for linear svm
print('Plot of no of support vectors versus c...\t')

c = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
fig, ax4 = plt.subplots(figsize=(8, 6), tight_layout=True)
ax4.semilogx(c, n_support_vectors, color='r', marker='o', markerfacecolor='m')

ax4.set_ylabel(f'no of support vectors', color='r')
ax4.set_xlabel(f'c')
ax4.set_xlim([1e-4, 1e4])
ax4.set_ylim(2300, 4000)
ax4.set_title(f"Number of support vectors", color='k', weight='normal', size=10)

plt.savefig("../figures/linear_support_vectors.jpg")
print('Done.\n')

## Part 2: Quadratic SVM
def quadratic_svm(c, bog, label):
    clf = svm.SVC(kernel='poly', C=c, degree=2, coef0=10)
    clf.fit(bog, label)
    bog_array = bog.toarray()
    y_pred = clf.predict(bog_array)
    acc = accuracy_score(y_pred, label)
    print(f"Accuracy on the quadratic-svm train set with C value {c}: {acc:.4f}", "\t")
    return clf, acc

###### predict using the trained quadratic classifier and tune the values of the hyperparameter c
c = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
training_acc = []
validation_acc = []
n_support_vectors_quad = []
for i in c:
    quad_clf, train_acc = quadratic_svm(pow(10, i), tfidf_mat_train, y_train)
    quad_ypred = quad_clf.predict(X_test)
    quad_acc = accuracy_score(quad_ypred, y_dev)
    training_acc.append(train_acc)
    validation_acc.append(quad_acc)
    n_support_vectors_quad.append(sum(quad_clf.n_support_))
    print(f"Accuracy on the quadratic-svm test set with C value 10^{i}: {quad_acc:.4f}")
    print("---------------------------------------------")

print("\n")

###### plot of the training and validation accuracy for quadratic svm
print('Plot of training versus validation accuracy...\t')

c = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
fig, ax3 = plt.subplots(figsize=(8, 6), tight_layout=True)
ax3.semilogx(c, training_acc, color='r', marker='o', markerfacecolor='m')

ax3.semilogx(c, validation_acc, color='b', marker='x', markerfacecolor='r')

ax3.set_ylabel(f'accuracy', color='r')
ax3.set_xlabel(f'c')
ax3.set_xlim([1e-4, 1e4])
ax3.set_ylim(0.78, 1.01)
ax3.set_title(f"Classification Accuracy", color='k', weight='normal', size=10)
ax3.legend(["training", "validation"], loc="upper left")

plt.savefig("../figures/quad_train_dev_acc_cmp.jpg")
print('Done.\n')

print("\n")

###### plot of the number of support vectors versus the regularization parameters, c for quadratic svm
print('Plot of number of support vectors verus c...\t')

c = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
fig, ax4 = plt.subplots(figsize=(8, 6), tight_layout=True)
ax4.semilogx(c, n_support_vectors_quad, color='r', marker='o', markerfacecolor='m')

ax4.set_ylabel(f'no of support vectors', color='r')
ax4.set_xlabel(f'c')
ax4.set_xlim([1e-4, 1e4])
ax4.set_ylim(2300, 4000)
ax4.set_title(f"Number of support vectors", color='k', weight='normal', size=10)

plt.savefig("../figures/quad_support_vectors.jpg")
print('Done.\n')

print("\n")

## Part 3: SVM with RBF kernel
def rbf_svm(c, bog, label, gamma_val='scale'):
    clf = svm.SVC(kernel='rbf', C=c, gamma=gamma_val)
    clf.fit(bog, label)
    bog_array = bog.toarray()
    y_pred = clf.predict(bog_array)
    acc = accuracy_score(y_pred, label)
    print(f"Accuracy on the rbf-svm train set with C value {c} and gamma value {gamma_val}: {acc:.4f}", "\t")
    return clf, acc

###### predict using the trained rbf classifier and tune the values of the hyperparameters c and \gamma
c = [10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3, 10e4]
gamma_values = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1]
training_acc = []
validation_acc = []
n_support_vectors = []

for i in c:
    for gamma in gamma_values:
        rbf_clf, train_acc = rbf_svm(i, tfidf_mat_train, y_train, gamma)
        rbf_ypred = rbf_clf.predict(X_test)
        rbf_acc = accuracy_score(rbf_ypred, y_dev)
        training_acc.append(train_acc)
        validation_acc.append(rbf_acc)
        n_support_vectors.append(sum(rbf_clf.n_support_))
        print(f"Accuracy on the rbf-svm test set with C value {i} and gamma value {gamma}: {rbf_acc:.4f}")
        print("---------------------------------------------")

print("\n")

training_accuracy = np.array(training_acc)
training_accuracy = np.reshape(training_accuracy, (len(c), len(gamma_values)))
validation_accuracy = np.array(validation_acc)
validation_accuracy = np.reshape(validation_accuracy, (len(c), len(gamma_values)))

###### plot of the training and validation accuracy for rbf svm
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

###### rbf training accuracy plot
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(
    training_accuracy,
    interpolation="nearest",
    cmap=plt.cm.hot,
    norm=MidpointNormalize(vmin=0.2, midpoint=0.92),
)
plt.xlabel("gamma")
plt.ylabel("C")
plt.colorbar()
plt.xticks(np.arange(len(gamma_values)), gamma_values, rotation=45)
plt.yticks(np.arange(len(c)), c)
plt.title("Training accuracy")
plt.savefig("../figures/training_acc_rbf.jpg")
print('Done.\n')
plt.show()

print("\n")

###### rbf validation accuracy plot
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(
    validation_accuracy,
    interpolation="nearest",
    cmap=plt.cm.hot,
    norm=MidpointNormalize(vmin=0.2, midpoint=0.92),
)
plt.xlabel("gamma")
plt.ylabel("C")
plt.colorbar()
plt.xticks(np.arange(len(gamma_values)), gamma_values, rotation=45)
plt.yticks(np.arange(len(c)), c)
plt.title("Validation accuracy")
plt.savefig("../figures/validation_acc_rbf.jpg")
print('Done.\n')
plt.show()

print("\n")

###### plot of the number of support vectors versus the regularization parameters, c for rbf kernel svm
c = [10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3, 10e4]
n_support_vectors_rbf = []


####### plot of support vectors for fixed lambda
for i in c:
    rbf_clf, train_acc = rbf_svm(i, tfidf_mat_train, y_train, 0.1)
    rbf_ypred = rbf_clf.predict(X_test)
    rbf_acc = accuracy_score(rbf_ypred, y_dev)
    n_support_vectors_rbf.append(sum(rbf_clf.n_support_))
        
c = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
fig, ax4 = plt.subplots(figsize=(8, 6), tight_layout=True)
ax4.semilogx(c, n_support_vectors_rbf, color='r', marker='o', markerfacecolor='m')

ax4.set_ylabel(f'no of support vectors', color='r')
ax4.set_xlabel(f'c')
ax4.set_xlim([1e-4, 1e4])
ax4.set_ylim(2300, 4000)
ax4.set_title(f"Number of support vectors for fixed gamma", color='k', weight='normal', size=10)

plt.savefig("../figures/rbf_svs_fixedlambda.jpg")
print('Done.\n')

print("\n")

####### plot of support vectors for fixed c
gamma_values = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1]
n_support_vectors_rbf2 = []

for gamma in gamma_values:
    rbf_clf, train_acc = rbf_svm(10, tfidf_mat_train, y_train, gamma)
    rbf_ypred = rbf_clf.predict(X_test)
    rbf_acc = accuracy_score(rbf_ypred, y_dev)
    n_support_vectors_rbf2.append(sum(rbf_clf.n_support_))
        
ga = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
fig, ax4 = plt.subplots(figsize=(8, 6), tight_layout=True)
ax4.semilogx(ga, n_support_vectors_rbf2, color='r', marker='o', markerfacecolor='m')

ax4.set_ylabel(f'no of support vectors', color='r')
ax4.set_xlabel(f'gamma')
ax4.set_xlim([1e-5, 1e1])
ax4.set_ylim(300, 9000)
ax4.set_title(f"Number of support vectors for fixed c", color='k', weight='normal', size=10)

plt.savefig("../figures/rbf_svs_fixedc.jpg")
print('Done.\n')

