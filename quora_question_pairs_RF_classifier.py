import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import gc
import itertools
import csv



header=[]
for i in range(128):
 header.append(i)

flag=[]
create=[]

print('excecution has started')

def text_vect_tfidf(tfidf):
    lis=[]
    lis.append(header)
    cx = scipy.sparse.coo_matrix(tfidf)
    temp_list = []
    temp = 0
    for i, j, v in zip(cx.row, cx.col, cx.data):
        # print( "(%d, %d), %s" % (i,j,v))
        temp_list.append(v)
        if temp != i:
            lis.append(temp_list[0:(len(temp_list) - 1)])
            temp_list = temp_list[(len(temp_list) - 1):]
        temp = i
    df=pd.DataFrame(lis,columns=header)
    return df




'''loading csv'''
data=pd.read_csv("train.csv")


vec=TfidfVectorizer()
label=data['is_duplicate'].copy()
data=data.drop('is_duplicate',1)

data['question1']=data['question1'].values.astype('U')
data['question1']=np.nan_to_num(data['question1'])
vec.fit_transform(raw_documents=data['question1'].values.astype('U'))
tfidf1=vec.transform(raw_documents=data['question1'],copy=True)
question1=text_vect_tfidf(tfidf1)



'''lis_fit=vec.get_feature_names()
test=[65736, 60310, 57698, 42980, 42229, 34203, 34200, 31890, 18290]
for i in test:
 print(lis_fit[i])'''

data['question2']=data['question2'].values.astype('U')
data['question2']=np.nan_to_num(data['question2'])
vec.fit_transform(raw_documents=data['question1'].values.astype('U'))
tfidf2=vec.transform(raw_documents=data['question1'],copy=True)
question2=text_vect_tfidf(tfidf2)



data=data.drop('question1',1)
data=data.drop('question2',1)
data=data.drop('qid1',1)
data=data.drop('qid2',1)


frames=[data,question1,question2]

data=pd.concat(frames,axis=1)


data=data.as_matrix()
data=np.nan_to_num(data)



X_train, X_test, y_train, y_test = train_test_split(data,label, test_size=0.33, random_state=42)
gnb = RandomForestClassifier(max_depth=128, random_state=1)
model = gnb.fit(data,label)
preds = gnb.predict(X_test)
print('accuracy is',accuracy_score(y_test,preds))

gc.collect()
gc.collect()

filename="test.csv"

test_preds=[]
test=[]
chunksize = 100000
for test_data in pd.read_csv(filename, chunksize=chunksize):
    test_data['question1'] = test_data['question1'].values.astype('U')
    test_data['question1'] = np.nan_to_num(test_data['question1'])
    vec.fit_transform(raw_documents=test_data['question1'].values.astype('U'))
    test_tfidf1 = vec.transform(raw_documents=test_data['question1'], copy=True)
    test_question1 = text_vect_tfidf(test_tfidf1)

    test_data['question2'] = test_data['question2'].values.astype('U')
    test_data['question2'] = np.nan_to_num(test_data['question2'])
    vec.fit_transform(raw_documents=test_data['question1'].values.astype('U'))
    test_tfidf2 = vec.transform(raw_documents=test_data['question1'], copy=True)
    test_question2 = text_vect_tfidf(test_tfidf2)

    test_data = test_data.drop('question1', 1)
    test_data = test_data.drop('question2', 1)

    test_frames = [test_data, test_question1, test_question2]
    test_data = pd.concat(test_frames, axis=1)


    test_data = test_data.as_matrix()
    test_data = np.nan_to_num(test_data)

    test_preds.append( list(gnb.predict(test_data)))
    print('epoch:',len(test_preds))



test_preds = list(itertools.chain.from_iterable(test_preds))
print(test_preds)
test_preds=test_preds[0:2345796]


test_data=pd.read_csv("test.csv")
test_data=test_data.drop('question1',1)
test_data=test_data.drop('question2',1)
result_header=['test_id']
df = pd.DataFrame(test_preds, columns=result_header)
result_frame=[test_data,df]
result_frame=pd.concat(result_frame,axis=1)
print(result_frame)
with open('C:/Users/Joe_John/Desktop/Academics/result1.csv', 'w+') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(['test_id','is_duplicate'])
    pass
    pass
    result_frame.to_csv(outcsv, index=False,header=False)
    pass
pass



