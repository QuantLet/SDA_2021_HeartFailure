[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **SDA_2021_HeartFailure** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of QuantLet: SDA_2021_HeartFailure

Published in: SDA_2021_St_Gallen

Description: 'This project aims to predict the death of patients suffering from heart disease. In this way, it might be possible to adapt the treatments and maybe avoid heart failures in some cases.'

Keywords: 'heart failure, death prediction, EDA, logistic regression, Decision classification tree, Random Forest Feature Selection, Test Accuracy'

Authors: 'Ozokcu Arzu, Therry Leonore'

Submitted: '07.12.2021'

Additional Info: 'This repository look at the feastures that explained the most the death of a patient suffering from heart disease. First there is an explanatory data analysis with distribution of the features and their correlation. Then, different models have been implemented : logistic regression, Decision classification tree, Random Forest Feature Selection and Test Accuracy'

```

### PYTHON Code
```python

#!/usr/bin/env python
# coding: utf-8

# In[1]:


#logistic regression

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing

from sklearn.metrics import accuracy_score as acc_rate

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.datasets import load_digits

from sklearn.tree import plot_tree

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import seaborn as sns
import sklearn

import plotly as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, classification_report, roc_curve,precision_recall_curve, auc,confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix

data = pd.read_csv("/Users/leonoretherry/Documents/St Gallen M2/smart data analytics/heart_failure_clinical_records_dataset.csv")

x = data.loc[:,:"time"]
y = data.loc[:,["DEATH_EVENT"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 2)


# In[2]:


#let's have an overview of our data set 
data.head()


# In[3]:


#Number and percentage of participants suffering from heart diseases in the study that died

data['DEATH_EVENT'].value_counts()

data['DEATH_EVENT'].value_counts(normalize=True)*100

#67% of the patient followed during the study did not die while 32 did


# In[4]:


data.isnull().sum()
#there are no missing values in our data set


# In[5]:



#overview of our explicative variables
hist = data.hist(figsize=(10,9))

plt.savefig("pandas_hist_01.png", bbox_inches='tight', dpi=100)


# In[6]:


#Let's remove the time component of our data set as it is not very representative for explaining the death event as it corresponds to the time period where the patient were followed by the study 

data2 = data.drop(['time'], axis = 1)

x2 = data.loc[:,:"smoking"]
y2 = data.loc[:,["DEATH_EVENT"]]

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size = 0.33, random_state = 2)

plt.figure(figsize=(15,8))
sns.heatmap(data2.corr(), annot=True)

#time, ejection fraction and serum creatinine are the more correlated variables with the death event
#A lower level of ejection fracion increase the chance to die while a higher level of serum creatinine increase the chance to die
#the time is negatively correlated with the death. It can be explained by the fact that the less time the patient has been followed by the study, the less chances he would have die. However it is possible that the patient die just after the end of the study


# In[7]:


#logistic regression

from sklearn.linear_model import LogisticRegression

scaler = preprocessing.StandardScaler().fit(x2_train)
x2_train_scaled = scaler.transform(x2_train)

model = LogisticRegression()

model.fit(x2_train_scaled, y2_train)

import statsmodels.api as sm
logit_model=sm.Logit(y2,x2)
result=logit_model.fit()
print(result.summary())

#time, ejection fraction and serum creatinine are once again the variables that have the greatest impact


# In[8]:


from sklearn import tree

#we kept max_depth=2 as the Accuracy rate was better with 2 than 3 or 4
clf2 = tree.DecisionTreeClassifier(criterion='entropy',
                                  max_depth=2)
clf2.fit(X=x2_train, y=y2_train)

# make prediction
print('\nThe target test data set is:\n', y2_test)
print('\nThe predicted result is:\n', clf2.predict(x2_test))
print('\nAccuracy rate is:\n', acc_rate(y2_test, clf2.predict(x2_test)))

plt.figure(figsize=(25,10))
c=plot_tree(clf2, 
            feature_names= x2.columns,
            class_names=['0','1'],
            filled=True, 
            rounded=True, 
            fontsize=14)
plt.savefig("decisiontreeOptimizedwithouttime.png")


# In[9]:


y2_pred=clf2.predict(x2_test)

print(confusion_matrix(y2_test,y2_pred))
target_names=['class 0', 'class 1']
print(classification_report(y2_test, y2_pred, target_names=target_names))

#the model better predict when the patient is going to live rather going to die 


# In[ ]:





# In[10]:


#Distribution of death event according to gender
len_data = len(data)
len_w = len(data[data["sex"]==0])
len_m = len_data - len_w

men_died = len(data.loc[(data["DEATH_EVENT"]==1) &(data['sex']==0)])
men_survived = len_m - men_died

women_died = len(data.loc[(data["DEATH_EVENT"]==1) & (data['sex']==1)])
women_survived = len_w - women_died

labels = ['Men died','Men survived','Women died','Women survived']
values = [men_died, men_survived, women_died, women_survived]

fig = go.Figure(data=[go.Pie(labels=labels, values=values,textinfo='label+percent',hole=0.4)])
fig.update_layout(
    title_text="Distribution of DEATH EVENT according to their gender")
fig.show()


# In[11]:


# Age distribution plot
fg=sns.FacetGrid(data, hue="DEATH_EVENT", height=6,)
fg.map(sns.kdeplot, "age",shade=True).add_legend(labels=["Alive","Not alive"])
plt.title('Age Distribution Plot');
plt.show()


# In[12]:


# Death event as per diabetes
pd.crosstab(data.diabetes ,data.DEATH_EVENT).plot(kind='bar')
plt.legend(title='DEATH_EVENT', loc='upper right', labels=['No death event', 'Death event'])
plt.title('Death Event as per diabetes ')
plt.xlabel('diabetes ')
plt.ylabel('# Death')
plt.show()


# In[13]:


# Death event as per high pressure blood
pd.crosstab(data.high_blood_pressure ,data.DEATH_EVENT).plot(kind='bar')
plt.legend(title='DEATH_EVENT', loc='upper right', labels=['Not alive', 'Alive'])
plt.title('Death Event as per High pressure blood ')
plt.xlabel('High pressure blood ')
plt.ylabel('# Death')
plt.show()


# In[14]:


#Death event as per smokers
pd.crosstab(data.smoking ,data.DEATH_EVENT).plot(kind='bar')
plt.legend(title='DEATH_EVENT', loc='upper right', labels=['Not alive', 'Alive'])
plt.title('Death Event as per smokers ')
plt.xlabel('Smokers ')
plt.ylabel('# Death')
plt.show()


# In[15]:


#Distribution of diabetics according to their gender
len_data = len(data)
len_w = len(data[data["sex"]==0])
len_m = len_data - len_w

men_with_diabetes = len(data.loc[(data["diabetes"]==1) & (data['sex']==1)])
men_without_diabetes = len_m - men_with_diabetes

women_with_diabetes = len(data.loc[(data["diabetes"]==1) & (data['sex']==0)])
women_without_diabetes = len_w - women_with_diabetes
labels = ['M_diabetes','M_no_diabete','W_diabete','W_no_diabete']
values = [men_with_diabetes, men_without_diabetes, women_with_diabetes, women_without_diabetes]

fig = go.Figure(data=[go.Pie(labels=labels, values=values,textinfo='label+percent',hole=0.4)])
fig.update_layout(
    title_text="Distribution of No/diabetics according to their gender. (M for Men, W for Women)")
fig.show()


# In[16]:


#Feature Selection according to their importance
x = data.copy()
y = x.loc[:,["DEATH_EVENT"]]
x = x.drop(columns=['time','DEATH_EVENT'])
features_names = x.columns
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(x, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# In[17]:


plt.figure()
plt.title("Feature importances")
sns.barplot(x=features_names[indices].to_numpy(), y=importances[indices], palette="deep",yerr=std[indices])
plt.xticks(range(x.shape[1]), features_names[indices].to_numpy(),rotation=80)
plt.xlim([-1, x.shape[1]])
plt.show()


# In[18]:


def plot_cm(cm,title):
    z = cm
    x = ['No death Event', 'Death Event']
    y = x
    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='deep')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix {}</b></i>'.format(title),
                      #xaxis = dict(title='x'),
                      #yaxis = dict(title='x')
                     )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.10,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.15,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=20),width=750,height=750)

    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()


# In[19]:


# Testing our models' accuracy rate

models= [['Logistic Regression ',LogisticRegression()],
        ['KNearest Neighbor ',KNeighborsClassifier()],
        ['Decision Tree Classifier ',DecisionTreeClassifier()],
        ['SVM ',SVC()]]

x = data.loc[:,:"time"]
y = data.loc[:,["DEATH_EVENT"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 2)

models_score = []
for (name,model) in models:
        model = model
        model.fit(x_train,y_train)
        model_pred = model.predict(x_test)
        cm_model = confusion_matrix(y_test, model_pred)
        models_score.append(accuracy_score(y_test,model.predict(x_test)))

        print(name)
        print('Validation Acuuracy: ',accuracy_score(y_test,model.predict(x_test)))
        print('Training Accuracy: ',accuracy_score(y_train,model.predict(x_train)))
        print('############################################')
        plot_cm(cm_model,title=name+"model")


# In[ ]:





```

automatically created on 2023-02-18