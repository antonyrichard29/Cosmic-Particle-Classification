import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
# Replace 'your_path_here.csv' with the correct path to your dataset
tele_data=pd.read_csv('your_path_here')
tele_data.describe()

# Visualize each feature for gamma and hadron
for label in (tele_data.columns):
  plt.hist(tele_data[tele_data['class']=='g'][label],color='green',label='gamma',alpha=0.5,density=True)
  plt.hist(tele_data[tele_data['class']=='h'][label],color='yellow',label='hadron',alpha=0.5,density=True)
  plt.title(label)
  plt.ylabel('Probability')
  plt.xlabel(label)
  plt.legend()
  plt.show()

# Train-Test Split
data_train,data_test=train_test_split(tele_data,test_size=0.2,random_state=42)

# Separate input and output
input_data_train=data_train.drop(columns=['class'])
output_data_train=data_train['class']
input_data_test=data_test.drop(columns=['class'])
output_data_test=data_test['class']

# Feature Scaling
scaler=StandardScaler()
input_data_train=scaler.fit_transform(input_data_train)
input_data_test=scaler.transform(input_data_test)

# Handle class imbalance
r_sam=RandomOverSampler()
input_data_train,output_data_train=r_sam.fit_resample(input_data_train,output_data_train)

# Train KNN Classifier
knn_model=KNeighborsClassifier(n_neighbors=1)
knn_model.fit(input_data_train,output_data_train)

# Predict and evaluate the model
out_pred=knn_model.predict(input_data_test)
accuracy_info=accuracy_score(output_data_test,out_pred)
print('\nThe Accuracy Score is :',accuracy_info)

print('\n--------------- Classification Report ---------------\n')
print(classification_report(output_data_test,out_pred))
