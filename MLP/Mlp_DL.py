from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from sklearn.metrics import mean_absolute_error
import pandas as pd

df=pd.read_csv("ad_org_train.csv")
#print df.head()
col=df["category"]
col2=df["likes"]
col3=df["dislikes"]
col4=df["comment"]
col5=df["views"]
#print col8
df=df.drop(["category"],axis=1)
df=df.drop(["likes"],axis=1)
df=df.drop(["dislikes"],axis=1)
df=df.drop(["comment"],axis=1)
df=df.drop(["views"],axis=1)
#print df.head()
for r in range (len(col)):
    #print r
    if(col[r]=='A'):
        col[r]=1
    elif(col[r]=='B'):
        col[r]=2
    elif(col[r]=='C'):
        col[r]=3
    elif(col[r]=='D'):
        col[r]=4
    elif(col[r]=='E'):
        col[r]=5
    elif(col[r]=='F'):
        col[r]=6
    elif(col[r]=='G'):
        col[r]=7
    elif(col[r]=='H'):
        col[r]=8

df["category"]=col
for r in range (len(col2)):
    if(col2[r]=='F'):
        col2[r]=0
for r in range (len(col3)):
    if(col3[r]=='F'):
        col3[r]=0
for r in range (len(col4)):
    if(col4[r]=='F'):
        col4[r]=0
for r in range (len(col5)):
    if(col5[r]=='F'):
        col5[r]=0
df['likes']=col2
df['dislikes']=col3
df['comment']=col4
df['views']=col5
#Code to convert duration of video into seconds.
j = df['duration']
time1=[]
a=2
for i in j:
    #print("for "+str(a))
    a=a+1
    hour ="0"
    minu="0"
    sec="0"
    mn =i.split("H")
    if len(mn)==2:
        #print "A"
        te = mn[0]
        hour=te[2:]
        #print(hour)
        ln = mn[1].split("M")
        if len(ln)==2:
            minu= ln[0]
            #print(minu)
            kr= ln[1].split("S")
            if len(kr)==2:
                sec = kr[0]
                #print(sec)
        if len(ln)==1:
            #print(minu)
            kr=ln[0].split("S")
            if len(kr)==2:
                sec = kr[0]
                #print(sec)
    if len(mn)==1:
        #print "B"
        ln = mn[0].split("M")
        #print(hour)
        if len(ln)==2:
            #print ln
            tem= ln[0]
            minu=tem[2:]
            #print(minu,"minu= ")
            kr= ln[1].split("S")
            if len(kr)==2:
                sec = kr[0]
                #print(sec)
        if len(ln)==1:
            #print(minu)
            kr=ln[0].split("S")
            sec = kr[0]
            #print(sec,"sec= ")
            if len(kr)==2:
                sec = kr[0]
                #print(sec)
            sec=sec[2:]
    t = float(hour)*3600+float(minu)*60+float(sec)
    #print t
    time1.append(t)
#
#
#
#
df["duration"]=time1
#df=df.drop(["vidid"],axis=1)
df=df.drop(["published"],axis=1)
#print df.head()

df2=pd.read_csv("ad_org_test.csv")
col=df2["category"]
col2=df2["likes"]
col3=df2["dislikes"]
col4=df2["comment"]
col5=df2["views"]
#print col8
df2=df2.drop(["category"],axis=1)
df2=df2.drop(["likes"],axis=1)
df2=df2.drop(["dislikes"],axis=1)
df2=df2.drop(["comment"],axis=1)
df2=df2.drop(["views"],axis=1)
#print df.head()
for r in range (len(col)):
    #print r
    if(col[r]=='A'):
        col[r]=1
    elif(col[r]=='B'):
        col[r]=2
    elif(col[r]=='C'):
        col[r]=3
    elif(col[r]=='D'):
        col[r]=4
    elif(col[r]=='E'):
        col[r]=5
    elif(col[r]=='F'):
        col[r]=6
    elif(col[r]=='G'):
        col[r]=7
    elif(col[r]=='H'):
        col[r]=8

df2["category"]=col
for r in range (len(col2)):
    if(col2[r]=='F'):
        col2[r]=0
for r in range (len(col3)):
    if(col3[r]=='F'):
        col3[r]=0
for r in range (len(col4)):
    if(col4[r]=='F'):
        col4[r]=0
for r in range (len(col5)):
    if(col5[r]=='F'):
        col5[r]=0
df2['likes']=col2
df2['dislikes']=col3
df2['comment']=col4
df2['views']=col5
#Code to convert duration of video into seconds.
j = df2['duration']
time1=[]
a=2
for i in j:
    #print("for "+str(a))
    a=a+1
    hour ="0"
    minu="0"
    sec="0"
    mn =i.split("H")
    if len(mn)==2:
        #print "A"
        te = mn[0]
        hour=te[2:]
        #print(hour)
        ln = mn[1].split("M")
        if len(ln)==2:
            minu= ln[0]
            #print(minu)
            kr= ln[1].split("S")
            if len(kr)==2:
                sec = kr[0]
                #print(sec)
        if len(ln)==1:
            #print(minu)
            kr=ln[0].split("S")
            if len(kr)==2:
                sec = kr[0]
                #print(sec)
    if len(mn)==1:
        #print "B"
        ln = mn[0].split("M")
        #print(hour)
        if len(ln)==2:
            #print ln
            tem= ln[0]
            minu=tem[2:]
            #print(minu,"minu= ")
            kr= ln[1].split("S")
            if len(kr)==2:
                sec = kr[0]
                #print(sec)
        if len(ln)==1:
            #print(minu)
            kr=ln[0].split("S")
            sec = kr[0]
            #print(sec,"sec= ")
            if len(kr)==2:
                sec = kr[0]
                #print(sec)
            sec=sec[2:]
    t = float(hour)*3600+float(minu)*60+float(sec)
    #print t
    time1.append(t)
#
#
#
#
df2["duration"]=time1
#df=df.drop(["vidid"],axis=1)
df2=df2.drop(["published"],axis=1)
#print df2.head()

x_train = df.drop('adview', axis=1)
x_train=x_train.drop('vidid',axis=1)
y_train = df['adview']

x_test = df2.drop('vidid', axis = 1)
#y_test = test['adview']
y_test_id=df2['vidid']

#build our model
def baseline_model():	
	model = Sequential()
	model.add(Dense(60,input_dim=6,activation='linear'))
	model.add(LeakyReLU(alpha=0.08))
	model.add(Dropout(0.25))
	model.add(Dense(50,activation='linear'))
	model.add(LeakyReLU(alpha=0.08))
	model.add(BatchNormalization(axis=-1))
	model.add(Dense(40,activation='linear'))
	model.add(LeakyReLU(alpha=0.08))
	model.add(Dropout(0.5))
	model.add(Dense(40,activation='linear'))
	model.add(LeakyReLU(alpha=0.08))
	model.add(Dropout(0.5))
	model.add(Dense(40,activation='linear'))
	model.add(LeakyReLU(alpha=0.08))
	model.add(Dropout(0.5))
	model.add(Dense(40,activation='linear'))
	model.add(LeakyReLU(alpha=0.08))
	model.add(Dropout(0.5))
	model.add(Dense(100,activation='linear'))
	model.add(LeakyReLU(alpha=0.08))
	model.add(Dropout(0.5))
	model.add(Dense(80,activation='linear'))
	model.add(LeakyReLU(alpha=0.08))
	model.add(Dropout(0.5))
	model.add(Dense(70,activation='linear'))
	model.add(LeakyReLU(alpha=0.08))
	model.add(Dropout(0.5))
	model.add(Dense(35,activation='linear'))
	model.add(LeakyReLU(alpha=0.08))
	model.add(Dropout(0.5))
	model.add(BatchNormalization(axis=-1))
	model.add(Dense(40,activation='linear'))
	model.add(LeakyReLU(alpha=0.08))
	model.add(Dropout(0.5))
	model.add(Dense(26,activation='linear'))
	model.add(LeakyReLU(alpha=0.08))
	model.add(Dropout(0.5))
	model.add(Dense(26,activation='linear'))
	model.add(LeakyReLU(alpha=0.08))
	model.add(Dropout(0.5))
	model.add(Dense(40,activation='linear'))
	model.add(LeakyReLU(alpha=0.08))
	model.add(Dropout(0.5))
	model.add(Dense(1,activation='linear'))
	model.compile(loss='mean_squared_error', optimizer='adam')
	#monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
	#model.fit(x,y,validation_data=(x_test,y_test),callbacks=[monitor],verbose=2,epochs=1000)
	return model

estimator=KerasClassifier(build_fn=baseline_model,epochs=500,batch_size=5, verbose=2)
estimator.fit(x_train,y_train)
y_pred = estimator.predict(x_test)

sub =pd.DataFrame()
sub['vidid']=y_test_id
sub['Predicted']=y_pred
sub.to_csv('MLP2at500.csv',index=False)






