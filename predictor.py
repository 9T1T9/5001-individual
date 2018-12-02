import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

def generate_train(df,df2):
    train=df.drop(['id','penalty'],axis=1)
    train['n_jobs'].loc[train['n_jobs']==-1]=16
    corr=train.corr()['time']
    df2.drop(corr[np.abs(corr)<0.1].index,axis=1,inplace=True)
    # feature correction
    df2['n_jobs'].loc[df2['n_jobs']==-1]=16
    # feature construction
    df2['max_iter_n_samples']=df2['max_iter']*df2['n_samples']
    df2['n_samples_n_features']=df2['n_samples']*df2['n_features']
    df2['n_samples_n_jobs']=df2['n_samples']/df2['n_jobs']
    df2['max_iter_n_jobs']=df2['max_iter']/df2['n_jobs']
    df2['n_samples_n_features_max_iter']=df2['n_samples']*df2['max_iter']*df2['n_features']
    temp=df2.drop(['penalty','id'],axis=1)
    scaling=StandardScaler()
    poly=PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    temp=scaling.fit_transform(poly.fit_transform(temp))
    temp=pd.DataFrame(temp,columns=poly.get_feature_names())   
    temp['penalty']=df2['penalty']
    temp=pd.get_dummies(temp)
    return temp

df=pd.read_csv('/home/zyt/bdt/5001/all/train.csv')
df2=pd.read_csv('/home/zyt/bdt/5001/all/test.csv')
fusiondata=pd.concat([df.drop(['time'],axis=1),df2],ignore_index=True)
data=generate_train(df,fusiondata)
data2=data.iloc[400:500]
data=data.iloc[0:400]
target=np.log(df['time'])


model=keras.Sequential([
                       keras.layers.Dense(128, kernel_initializer='normal',input_dim=data.shape[1],activation=tf.nn.relu),
                       keras.layers.Dense(128,kernel_initializer='normal',activation=tf.nn.relu),
                       keras.layers.Dense(64, kernel_initializer='uniform',activation=tf.nn.relu),
                       keras.layers.Dense(1)
                                 ])
    
optimizer = tf.train.AdamOptimizer(0.001)
model.compile(loss='mse',optimizer=optimizer,metrics=['mse'])
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

while True:
    history=model.fit(data.values,target.values,validation_split=0.2,epochs=80,batch_size=4,verbose=0,shuffle='True')
    print(history.history['val_loss'])
    if history.history['val_loss'][-1]<0.2:
        break
#    print(np.median(history.history['val_loss']))

result=[i[0] for i in np.exp(model.predict(data2.values))]
#result=[i[0] for i in np.exp(np.mean([model.predict(data2.values) for model in models],axis=0))]        
tdata={'id':range(0,100),'time':result}
tdata=pd.DataFrame(tdata)
tdata.to_csv('predictions.csv',index=False)

