import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split

from sklearn.ensemble import RandomForestRegressor
import time
import re



df=pd.read_csv("/content/housePrice.csv")


new_df= df.drop(["Price"],axis=1)
new_df.head()
new_df.dropna(inplace=True)


training_dataset=new_df.drop("Price(USD)",axis=1)

labels=new_df["Price(USD)"]

training_dataset["Parking"]=training_dataset["Parking"].apply(lambda x:1 if x==True else 0)
training_dataset["Elevator"]=training_dataset["Elevator"].apply(lambda x:1 if x==True else 0)
training_dataset["Warehouse"]=training_dataset["Warehouse"].apply(lambda x:1 if x==True else 0)
training_dataset['Area'] = training_dataset['Area'].apply(lambda x: re.sub(',','',x))
address_dummy = pd.get_dummies(training_dataset['Address'])
training_dataset = training_dataset.merge(address_dummy,left_index=True,right_index=True)
training_dataset= training_dataset.drop(["Address"],axis=1)
labels=np.asarray(labels)
training_dataset= np.asarray(training_dataset)


x_train,x_test,y_train,y_test = train_test_split(training_dataset,labels,test_size=0.2)

model= RandomForestRegressor(n_estimators=150,max_depth=20)
model.fit(x_train,y_train)
class user:

  
  def __init__(self,Area,Room,Parking,Warehouse,Elevator,Address):
    self.Area=Area
    self.Room=Room
    self.Parking=Parking
    self.Warehouse=Warehouse
    self.Elevator=Elevator
    self.Address=Address



    
    

list1=["Area","Room","Parking","Warehouse","Elevator","Address"]
list2=[]
for i in range(6):
     if list1[i]!="Address":
      list2.append(int(input(f"import the rate of {list1[i]}: ")))
     else:
         list2.append(input(f"import the rate of {list1[i]}: "))







user_data=user(list2[0],list2[1],list2[2],list2[3],list2[4],list2[5])

dic={
   "Area": [list2[0]],
   "Room":[list2[1]],
   "Parking":[list2[2]],
    "Warehouse":[list2[3]],
    "Elevator":[list2[4]],
    "Address":[list2[5]],
}


user_dataframe=pd.DataFrame(data=dic)
user_dataframe=user_dataframe.merge(address_dummy,left_index=True,right_index=True)
user_dataframe=user_dataframe.drop(["Address"],axis=1)
user_dataframe=np.asarray(user_dataframe)
result= model.predict(user_dataframe)
accuracy=model.score(x_test,y_test)
for c in result:
      print(f"{int(c)} $ your target house costs in Tehran in (2021) but at {time.ctime(time.time())} it must be {int(c*1.55)} $")
           
print(f"accuracy score of the model : {accuracy}")