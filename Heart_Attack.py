import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time

import warnings
warnings.filterwarnings(action="ignore")

pd.set_option('display.width', 1000)
pd.set_option('display.max_column', 20)

print("\nML solution proposed by : Muskan Gupta")
print("EMAIL ID : muskangupta042002@gmail.com")
print("Student ID : 376095")

data = pd.read_csv('HeartAttack_data.csv', index_col=False)
print("\n\n\nSample HeartAttack dataset head(5) :- \n\n", data.head(5) )

print("\n\nShape of the HeartAttack dataset  data.shape = ", end="")
print( data.shape)

array=data.values
print("\nHeartAttack data decription : \n")
print( data.describe() )
data.replace('?',np.nan,inplace =True)
new_data=data.drop(['slope', 'ca', 'thal'], axis=1)
print(new_data)

datas = new_data.fillna(new_data.median())
print("After replacing ? with median :\n",datas)
print("Check data has any null values or not:\n",datas.isnull().sum())
y=pd.DataFrame(datas)
plt.hist(y['num'])
plt.show()

print("\n\n\ndata.groupby('num').size()\n")
print(y.groupby('num').size())

print(y.dtypes)

y['trestbps'] = y['trestbps'].astype(float)
y['chol'] = y['chol'].astype(float)
y['fbs'] = y['fbs'].astype(float)
y['restecg'] = y['restecg'].astype(float)
y['thalach'] = y['thalach'].astype(float)
y['exang'] = y['exang'].astype(float)
y['oldpeak'] = y['oldpeak'].astype(float)

print("Object to float tpye:\n ",(y.dtypes))

y.plot(kind='density', subplots=True, layout=(4,5), sharex=False, legend=False, fontsize=1)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
cax = ax1.matshow(y.corr(), vmin=-1, vmax=1 )
ax1.grid(True)
plt.title('Heart Attack Attributes Correlation')
# Add colorbar, make sure to specify tick locations to match desired ticklabels
fig.colorbar(cax)
plt.show()

Y = y['num'].values
x = y.drop('num', axis=1).values
test = SelectKBest(score_func=chi2, k=7) #K - how many best columns
fit = test.fit(x, Y)
X = fit.transform(x)
print(X.shape)
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.33, random_state=21)
models_list = []
models_list.append(('CART', DecisionTreeClassifier()))
models_list.append(('SVM', SVC()))
models_list.append(('NB', GaussianNB()))
models_list.append(('KNN', KNeighborsClassifier()))
models_list.append(('LDA', LinearDiscriminantAnalysis()))
num_folds = 10

results = []
names = []
for name, model in models_list:
    kfold = KFold(n_splits=num_folds, random_state=123)
    startTime = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    endTime = time.time()
    results.append(cv_results)
    names.append(name)
    print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), endTime-startTime))

fig = plt.figure()
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

pipelines=[]
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC( ))])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis( ))])))
results = []
names = []



print("\n\n\nAccuracies of algorithm after scaled dataset\n")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kfold = KFold(n_splits=num_folds, random_state=123)
    for name, model in pipelines:
        start = time.time()
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))

fig = plt.figure()
fig.suptitle('Performance Comparison after Scaled Data')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

model = GaussianNB()
start = time.time()
model.fit(X_train_scaled, Y_train)   #Training of algorithm
end = time.time()
print( "\n\nGaussianNB Training Completed. It's Run Time: %f" % (end-start))

X_test_scaled = scaler.transform(X_test)

predictions = model.predict(X_test_scaled)
print("All predictions done successfully by GaussianNB Machine Learning Algorithms")
print("\n\nAccuracy score %f" % accuracy_score(Y_test, predictions))

print("\n")
print("confusion_matrix = \n")
print( confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))


from sklearn.externals import joblib
filename =  "finalized_HeartAttack_model.sav"
joblib.dump(model, filename)
print( "Best Performing Model dumped successfully into a file by Joblib")


print("\nML solution proposed by : Muskan Gupta")
print("EMAIL ID : muskangupta042002@gmail.com")
print("Student ID : 376095")
