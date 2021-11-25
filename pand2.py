import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
data=pd.read_csv(r"C:\Users\Asus\AppData\Local\Programs\Python\Python38-32\diabetes.csv")
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print(data.head(5))
#print(rd.head(5))
#print(rd['Age'])
#print(rd.isnull())
#print(sns.distplot(rd.Age))
#print(plt.show())
#print(rd.shape)
#print(rd.describe())
#print(sns.countplot(x='Outcome',data=rd))
#print(plt.show())
#a=np.percentile(rd.Insulin,[99])[0]
#print(a)
#rd.Insulin[(rd.Insulin> 1.5*a)]=1.5*a
#print(rd[(rd.Insulin>a)])
#print(rd.Insulin.mean())
#print(rd.describe)
#print(rd.info())
#k=sn.add_constant(rd['Age'])
#print(x)
#ni=sn.OLS(rd['Outcome'],k).fit()
#print(ni.summary())
#y=rd['Outcome']
#z=rd[['DiabetesPedigreeFunction']]
#ni2=LinearRegression()
#ni2.fit(z,y)
#print(ni2.intercept_,ni2.coef_)
#print("\||||||||")
#print(ni2.predict([[0.623]]))
#print(ni2.predict(z))
#sns.jointplot(x=rd['DiabetesPedigreeFunction'],y=rd['Outcome'],data=rd,kind='reg')
print(data['Glucose'].max())
print(data.info())
print(data.shape)
print(data.describe())
print(data.isnull().sum())
print(sns.countplot(x="Outcome",data=data))
print(plt.show())
print(data["Outcome"].value_counts())
plt.figure(figsize=(16,5))
print(sns.heatmap(data.corr(),annot=True))
plt.title('Correlation Matrix (for diabetes prediction)')
plt.show()
from sklearn.model_selection import train_test_split
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=10)
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
lr_model = LogisticRegression()
lr_model.fit(X_train,y_train)

lr_prediction = lr_model.predict(X_test)
print('Logistic Regression accuracy = ', metrics.accuracy_score(lr_prediction,y_test))
svm_model = svm.SVC()
svm_model.fit(X_train,y_train)

svc_prediction = svm_model.predict(X_test)
print('SVM accuracy = ', metrics.accuracy_score(svc_prediction,y_test))
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)

dt_prediction = dt_model.predict(X_test)
print('Decision Tree accuracy = ', metrics.accuracy_score(dt_prediction,y_test))
knn_model = KNeighborsClassifier()
knn_model.fit(X_train,y_train)

knn_prediction =knn_model.predict(X_test)
print('KNN accuracy = ', metrics.accuracy_score(knn_prediction,y_test))
rf_model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_model.fit(X_train, y_train)

rf_prediction=rf_model.predict(X_test)
print('random forest accuracy = ',metrics.accuracy_score(rf_prediction,y_test))

pickle.dump(lr_model,open('dia.pkl','wb'))
dia=pickle.load(open('dia.pkl','rb'))
print(dia)
