import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
data=pd.read_csv('hypertension_data.csv')
print(data.info())
print(data.isna().sum())
print(data.columns)
print(data.sex)
print(data.max()-data.min())
col=data.columns.values
plt.plot(data['age'].head(100),marker='o',label='age(0-87)',color='red')
plt.plot(data['sex'].head(100),marker='o',label='sex(1-0)',color='blue')
plt.legend()
plt.show()
plt.plot(data['sex'].head(100),marker='o',label='sex(0-87)',color='red')
plt.plot(data['target'].head(100),marker='o',label='target(1-0)',color='blue')
plt.legend()
plt.show()
sn.pairplot(data)
plt.show()
for i in col:
    sn.boxplot(data[i])
    plt.show()
    if len(data[i].value_counts().values) <= 5:
        sn.countplot(data[i])
        plt.show()
data['z-scores']=(data.trestbps-data.trestbps.mean())/(data.trestbps.std())
df=data[(data['z-scores'] >-3) & (data['z-scores'] < 3)]
qa1=df.trestbps.quantile(0.25)
qa3=df.trestbps.quantile(0.75)
iqr=qa3-qa1
up=qa3+1.5*iqr
low=qa1-1.5*iqr
df=df[(df.trestbps <up)&(df.trestbps >low)]
qua1=df.oldpeak.quantile(0.25)
qua3=df.oldpeak.quantile(0.75)
Iqr=qua3-qua1
upper=qua3+1.5*Iqr
lower=qua1-1.5*Iqr
df=df[(df.oldpeak <upper)&(df.oldpeak >lower)]
print(data.trestbps.shape)
print(df.trestbps.shape)
print(df.oldpeak.shape)
sn.boxplot(df.trestbps)
plt.show()
sn.boxplot(df.chol)
plt.show()
quant1=df.oldpeak.quantile(0.25)
quant3=df.oldpeak.quantile(0.75)
IQr=quant3-quant1
upper_lim=quant3+1.5*Iqr
lower_lim=qua1-1.5*Iqr
df=df[(df.oldpeak <upper_lim)&(df.oldpeak >lower_lim)]
quanti1=df.chol.quantile(0.25)
quanti3=df.chol.quantile(0.75)
IQR=quanti3-quanti1
upper_limi=quanti3+1.5*IQR
lower_limi=quanti1-1.5*IQR
df=df[(df.chol <upper_limi)&(df.chol >lower_limi)]
df['sex']=df.sex.fillna(df.sex.mean())
print(df.shape)
print(df.columns)
print(df.info())
print(df.isna().sum())
print(df.describe())
x=df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y=df['target']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(x_train,y_train)
log_predict=lg.predict(x_test)
from sklearn.tree import DecisionTreeClassifier
dtree_classi=DecisionTreeClassifier()
dtree_classi.fit(x_train,y_train)
dtree_predict=dtree_classi.predict(x_test)
from sklearn.neighbors import KNeighborsClassifier
knn_classi=KNeighborsClassifier()
knn_classi.fit(x_train,y_train)
knn_pred=knn_classi.predict(x_test)
plt.plot(y_test,marker='o',color='red',label='y_test')
plt.plot(log_predict,marker='o',color='blue',label='logistic_reg_prediction')
plt.title('Logistic regression prediction vs Y_test')
plt.legend()
plt.show()
plt.plot(y_test,marker='o',color='red',label='y_test')
plt.plot(knn_pred,marker='o',color='blue',label='KNN_Classification_prediction')
plt.title('KNN Classification prediction vs Y_test')
plt.legend()
plt.show()
plt.plot(y_test,marker='o',color='red',label='y_test')
plt.plot(dtree_predict,marker='o',color='blue',label='decision_tree_prediction')
plt.title('Decision tree classification prediction vs Y_test')
plt.legend()
plt.show()
from keras.models import Sequential
from keras.layers import Dense
import keras.activations,keras.losses
models=Sequential()
models.add(Dense(units=x.shape[1],input_dim=x_train.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.relu))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=1,activation=keras.activations.sigmoid))
models.compile(optimizer='adam',metrics='accuracy',loss=keras.losses.binary_crossentropy)
hist=models.fit(x_train,y_train,batch_size=20,epochs=200,validation_split=0.45)
plt.plot(hist.history['accuracy'],label='training accuracy',marker='o',color='red')
plt.plot(hist.history['val_accuracy'],label='val_accuracy',marker='o',color='blue')
plt.title('Training Vs  Validation accuracy')
plt.legend()
plt.show()