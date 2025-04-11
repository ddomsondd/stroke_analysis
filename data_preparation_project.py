"""
The aim of this study is to examine the impact of various factors on whether or not a person experiences a stroke.
To do this, a data analysis will be conducted, the dataset will be appropriately prepared, and then models will be built to predict whether a patient will have a stroke (value in the 'stroke' column).

If you have any comments, feel free to reach out via Discord: dobosh#5559 or via email: s97583@pollub.edu.pl / dominika.dobosz23@gmail.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
I started the task by loading the dataset, displaying its columns, and showing the first few rows.
We have an 'id' column which will definitely be removed, gender, age, whether the patient has hypertension or heart disease,
marital status, type of work, residence (urban or rural), average glucose level, BMI, smoking status, and finally whether they had a stroke.
"""
healthcare = pd.read_csv('healthcare-dataset-stroke-data.csv', sep=",")
columns = healthcare.columns.to_list()
print("COLUMNS: ", columns)
healthcare.head()

# I removed the 'id' column as it doesn't contribute anything useful.
healthcare.drop(columns=['id'], inplace=True)

# For consistency and aesthetics, I changed the column name so that they all start with lowercase letters.
healthcare.rename(columns={'Residence_type': 'residence_type'}, inplace=True)

# I checked the dataset’s dimensions (number of rows and columns) and info (non-null values and data types).
print("Dataset shape: ", healthcare.shape)
healthcare.info()

# I checked how many missing values there are.
healthcare.isna().sum()
# As you can see, the missing values aren't extensive – the 'bmi' column is the most affected.

# I created lists of categorical and numerical columns.
categorical_columns = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'residence_type', 'smoking_status', 'stroke']
numerical_columns = ['age', 'avg_glucose_level', 'bmi']

# I generated statistics for numerical data while excluding missing values.
# I excluded the 'count' row as it’s not very meaningful for this context.
healthcare[numerical_columns].dropna().describe().drop('count')

"""
From the results, we can already spot a few issues – for example, the minimum age is 0.08.
  It’s not a negative number (which would obviously be wrong), but still quite questionable.
  Assuming 0.08 means a newborn – would such an individual be included in this study? Tough call – but I decided to leave it as is.
The maximum age of 82 seems fine.
The average glucose levels show a wide range and some extreme values (could be hypoglycemia or diabetes).
Same goes for BMI – 2.0 is definitely an anomaly, and 97.6 would indicate severe obesity.
"""

# Displaying unique values for categorical columns
for i in categorical_columns:
  print(f"Unique values for column {i}:\n", healthcare[i].unique())
  print(f"Value counts:\n", healthcare[i].value_counts(), "\n")

"""
For 'gender' – we have three options. Normally we could include 'Other' too,
  but since it’s only one sample, there’s no point keeping it – that observation will be removed.
'hypertension', 'heart_disease', and 'ever_married' are binary (yes/no).
For 'work_type' – we might consider merging some categories – e.g., 'children' and 'never_worked' as 'unemployed'.
'residence_type' has 28 missing values – not a lot, so instead of deleting, we could mark them as 'unknown' (to consider).
'smoking_status' has many classes too – a bit harder to merge logically.
Finally, 'stroke' has 2 classes (0 = no, 1 = yes) – that’s what we’re predicting.
"""

# Creating histograms for numerical columns
for i in numerical_columns:
  plt.hist(healthcare[i], color='purple', bins=20, edgecolor='black')
  plt.title(f"Histogram for {i}")
  plt.ylabel("Count")
  plt.xlabel("Value")
  plt.show()

"""
'age':
The values in the 'age' column seem plausible – no negative values.
Age around 80 being one of the most common ranges might raise some questions,
  but it’s logical considering this is a stroke-related study – the participants are likely to be older adults.
On the flip side, there are quite a few children in the dataset, which is a bit surprising,
  since stroke risk increases with age. But since strokes can occur at any age (including in children), I’ll assume the 'age' column is okay.
The distribution is not normal, but this could just reflect randomness in participant selection.
"""

"""
'avg_glucose_level':
This histogram is right-skewed and not symmetrical. The values range widely. According to general medical guidelines:

Fasting glucose:
    70-99 mg/dL – normal
    100-125 mg/dL – impaired glucose tolerance (pre-diabetes)
    >126 mg/dL – diabetes

Glucose after 120 minutes in an oral glucose tolerance test (OGTT):
    <140 mg/dL – normal
    140–199 mg/dL – impaired
    ≥200 mg/dL – diabetes

It’s unclear whether these measurements were taken fasting or after an OGTT.
The values range so widely that it seems both might have been the case.
Most values fall between 50 and 130 mg/dL, but a lot of patients have values over 200 – going up to 270 – which is very extreme.
"""

"""
'bmi':
BMI forms a roughly normal distribution, slightly right-skewed.
Obvious outliers pop up – both low and high – indicating underweight and extreme obesity.
These values will definitely be handled later.
"""

"""
I also checked how many children were under one year and under 18 (since 'children' is a category in 'work_type').
There are 854 patients under 18 and 687 labeled as 'children' in 'work_type', which is reasonable – maybe some older teens work part-time.
I checked how many patients had glucose levels below 70 or above 200 – that was a significant number (1183).
Similarly, many patients had BMI values under 18 or over 40 – another red flag.
"""

print("Number of children under 1 year old: ", healthcare[(healthcare['age'] < 1)].shape[0])
print("Number of minors (under 18): ", healthcare[(healthcare['age'] < 18)].shape[0])
print("Extreme glucose values: ", healthcare[(healthcare['avg_glucose_level'] < 70) | (healthcare['avg_glucose_level'] > 200)].shape[0])
print("Extreme BMI values: ", healthcare[(healthcare['bmi'] < 18) | (healthcare['bmi'] > 40)].shape[0])

"""
I created a correlation heatmap of numerical values with the target variable – 'stroke'.
Turns out the correlation is generally low, but 'age' has the strongest link to stroke occurrence, which is quite logical.
"""

cor_mat = healthcare[['age', 'avg_glucose_level', 'bmi', 'stroke']].dropna().corr().round(2)
plt.figure(figsize=(6,6))
sns.heatmap(cor_mat, vmin=-1, vmax=1, annot=True, cmap='coolwarm')
plt.title("Correlation matrix for stroke dataset")
plt.show()

"""
I also created boxplots for numerical columns.
No outliers in 'age'.
Plenty of outliers in 'avg_glucose_level', but I’ll focus on medical norms rather than pure statistical outliers.
Same for 'bmi'.
"""

for i in numerical_columns:
  plt.boxplot(healthcare[i].dropna())
  plt.title(f'Boxplot for {i}')
  plt.show()

"""
Here I removed the row with 'Other' gender and missing 'residence_type' (only 29 rows total).
Then I grouped some 'work_type' categories:
  - 'children' and 'Never_worked' -> 'unemployed'
  - 'Private' and 'Self-employed' -> 'non_govt_job'
  - Renamed 'Govt_job' to 'govt_job' for consistency
"""

print(healthcare[(healthcare['gender'] == 'Other') | ((healthcare['residence_type'].isna()))].shape[0])
healthcare = healthcare[(healthcare['gender'] != 'Other') & ((healthcare['residence_type'] == 'Urban') | (healthcare['residence_type'] == 'Rural'))]
healthcare['work_type'] = healthcare['work_type'].replace({
    'children': 'unemployed',
    'Never_worked': 'unemployed',
    'Govt_job': 'govt_job',
    'Private': 'non_govt_job',
    'Self-employed': 'non_govt_job'
})

healthcare.isna().sum()
healthcare.info()

# Detecting outliers using IQR method
from IPython.display import display
q1 = healthcare[numerical_columns].quantile(0.25)
q3 = healthcare[numerical_columns].quantile(0.75)
iqr = q3 - q1
low_bound = (q1 - 1.5 * iqr)
upp_bound = (q3 + 1.5 * iqr)
num_of_outliers_L = (healthcare[iqr.index] < low_bound).sum()
num_of_outliers_U = (healthcare[iqr.index] > upp_bound).sum()
outliers_15iqr = pd.DataFrame({'lower_boundary':low_bound, 'upper_boundary':upp_bound, 'num_of_outliers_L':num_of_outliers_L, 'num_of_outliers_U':num_of_outliers_U})
pd.set_option('display.max_columns', None)
display(outliers_15iqr)

"""
Checked number of extreme BMI and glucose values.
Even though the values are extreme (e.g., BMI 10 or 60), only 13 rows fall in that range – could be removed.
For glucose, 245 rows are extreme (~5% of dataset).
I considered replacing with mean or using KNN imputation, but it skewed distributions too much, so I opted to drop them.
"""

print("Extreme BMI: ", (healthcare[(healthcare['bmi'] < 10) | (healthcare['bmi'] > 60)].shape[0]))
print("Extreme glucose: ", (healthcare[(healthcare['avg_glucose_level'] < 60) | (healthcare['avg_glucose_level'] > 250)].shape[0]))
print("Normal BMI: ", healthcare[(healthcare['bmi'] >= 10) & (healthcare['bmi'] <= 60)].shape[0])
print("Normal glucose: ", healthcare[(healthcare['avg_glucose_level'] >= 60) & (healthcare['avg_glucose_level'] <= 250)].shape[0])

# Remove extreme values but retain missing (to be imputed later)
healthcare = healthcare[(healthcare['bmi'].isna()) | ((healthcare['bmi'] >= 10) & (healthcare['bmi'] <= 60))]
healthcare = healthcare[(healthcare['avg_glucose_level'].isna()) | ((healthcare['avg_glucose_level'] >= 60) & (healthcare['avg_glucose_level'] <= 250))]
print(healthcare.shape)
healthcare.info()

# Impute remaining NaNs with KNN
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
healthcare[numerical_columns] = scaler.fit_transform(healthcare[numerical_columns])

imputer = KNNImputer(n_neighbors=5)
healthcare[numerical_columns] = imputer.fit_transform(healthcare[numerical_columns])
healthcare[numerical_columns] = scaler.inverse_transform(healthcare[numerical_columns])

healthcare.isna().sum()

"""
Next, I did some feature engineering to increase correlation between features and the target variable.
Created:
  - 'is_elderly' -> whether the patient is over 60
  - 'has_risk_factors' -> 0, 1, or 2 depending on presence of hypertension and/or heart disease
  - 'bmi_category' -> underweight, normal, or overweight
  - 'glucose_binned' -> low, normal, or high glucose
"""

healthcare['is_elderly'] = (healthcare['age'] > 60).astype(int)
healthcare['has_risk_factors'] = healthcare['hypertension'] + healthcare['heart_disease']

def categorize_bmi(bmi):
    if bmi < 18.5:
        return 0   #'underweight'
    elif 18.5 <= bmi < 25:
        return 1   #'normal'
    elif 25 <= bmi < 30:
        return 2   #'overweight'
    else:
        return 3   #'obese'

healthcare['bmi_category'] = healthcare['bmi'].apply(categorize_bmi)

def bin_glucose(value):
    if value < 90:
        return 0  #'low'
    elif 90 <= value < 140:
        return 1  #'normal'
    else:
        return 2  #'high'

healthcare['glucose_binned'] = healthcare['avg_glucose_level'].apply(bin_glucose)

# Next, I displayed the statistics for the numerical columns after modifying their values.
healthcare[numerical_columns].describe().drop('count')

# I also displayed the first rows again to check how the dataset looks after adding the columns.
healthcare.head()

# I recreated histograms for the numerical columns.

for i in numerical_columns:
  plt.hist(healthcare[i], color='purple', bins=20, edgecolor='black')
  plt.title(f"Histogram for column {i}")
  plt.ylabel("Frequency")
  plt.xlabel("Value")
  plt.show()

"""
Next, I transformed categorical values using Label Encoding.
At the same time, I check how the values in the data columns have been encoded, 
as this will be needed for the next step.
"""

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le_columns = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
for i in le_columns:
  healthcare[i] = le.fit_transform(healthcare[i])
  print(dict(zip(le.classes_, le.transform(le.classes_))))

"""
#healthcare = pd.get_dummies(healthcare, columns=['bmi_category', 'glucose_binned'], drop_first=True)
bmi_order = {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3}
healthcare['bmi_category'] = healthcare['bmi_category'].map(bmi_order)

glucose_order = {'low': 0, 'normal': 1, 'high': 2}
healthcare['glucose_binned'] = healthcare['glucose_binned'].map(glucose_order)
"""

healthcare.head()

"""
Here, I wanted to check the data in relation to age—since the dataset includes many children, I wanted to ensure
that these observations make sense. Hypertension, heart diseases, and strokes occur much more frequently in adults. 
There are only isolated cases where someone under the age of 20 experienced any of these conditions, but these can happen at a young age, 
so the dataset still seems logical.
The most controversial columns could be 'ever_married', 'smoking_status', and 'work_type' in relation to 'age', 
but as seen in the boxplots, the data seems fairly logical.
Marriage occurred earliest for patients just before the age of 20, which, though rare today, could still theoretically happen, 
as in Poland, it is legal to get married at 18.
Regarding work type, it is clear that the majority of children are 'unemployed'. It might be concerning that some children around 
the age of 10 are working in the private sector, but considering that children can be employed as, for example, actors, 
this could be valid. Moreover, there are not many such cases, so it shouldn't raise significant concerns.
The 'smoking_status' column seems the most problematic—boxplots 0 and 2 correspond to 'unknown' and 'never smoked', so 
it can be assumed that a lack of knowledge means that the child did not smoke (hopefully). 
The graphs for 1 and 3, representing 'formerly smoked' and 'smokes', show that there are observations of children around 10 years old who smoke 
or have smoked in the past, but it's hard to tell if this is truly accurate data. Given that the boxplots for both cases are shifted upward, 
it seems these might be very isolated cases.
"""

sns.set(style="whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

#hypertension
sns.boxplot(x='hypertension', y='age', data=healthcare, ax=axes[0,0])
axes[0,0].set_title('Age vs Hypertension', fontweight='bold')
axes[0,0].set_xlabel('Hypertension (0 = no, 1 = yes)')
axes[0,0].set_ylabel('Age')

#heart disease
sns.boxplot(x='heart_disease', y='age', data=healthcare, ax=axes[0,1])
axes[0,1].set_title('Age vs Heart Disease', fontweight='bold')
axes[0,1].set_xlabel('Heart Disease (0 = no, 1 = yes)')
axes[0,1].set_ylabel('Age')

#ever married
sns.boxplot(x='ever_married', y='age', data=healthcare, ax=axes[1,0])
axes[1,0].set_title('Age vs Marriage', fontweight='bold')
axes[1,0].set_xlabel('Marital Status (0 = no, 1 = yes)')
axes[1,0].set_ylabel('Age')

#smoking status
sns.boxplot(x='smoking_status', y='age', data=healthcare, ax=axes[1,1])
axes[1,1].set_title('Age vs Smoking', fontweight='bold')
axes[1,1].set_xlabel('Smoking Status')
axes[1,1].set_ylabel('Age')
axes[1,1].tick_params(axis='x', rotation=30)

#work type
sns.boxplot(x='work_type', y='age', data=healthcare, ax=axes[2,0])
axes[2,0].set_title('Age vs Work Type', fontweight='bold')
axes[2,0].set_xlabel('Work Type')
axes[2,0].set_ylabel('Age')
axes[2,0].tick_params(axis='x', rotation=30)

#stroke
sns.boxplot(x='stroke', y='age', data=healthcare, ax=axes[2,1])
axes[2,1].set_title('Age vs Stroke', fontweight='bold')
axes[2,1].set_xlabel('Stroke (0 = no, 1 = yes)')
axes[2,1].set_ylabel('Age')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

"""
I created a Spearman correlation plot for categorical columns.
Unfortunately, it's noticeable that most correlations are low.
For the 'stroke' column, we can highlight hypertension, heart problems, and marital status (which is logical, 
as this column has a higher correlation with 'age', and 'age' has one of the highest correlations with 'stroke').
"""

corr_spearman = healthcare[categorical_columns].corr(method='spearman')
plt.figure(figsize=(10, 8))
sns.heatmap(corr_spearman, annot=True, cmap='Purples', square=True)
plt.title('Spearman Correlation (Categorical Features)')
plt.show()

"""
This is the correlation plot for numerical values.
In relation to the 'stroke' column, the 'age' column stands out, which makes sense because the risk of a stroke increases with age.
The 'average glucose level' column also shows a relatively high (for this standard) correlation.
"""
cor_mat = healthcare[['age', 'avg_glucose_level', 'bmi', 'stroke']].dropna().corr().round(2)
plt.figure(figsize=(6,6))
sns.heatmap(cor_mat, vmin=-1, vmax=1, annot=True, cmap='coolwarm')
plt.title("Pearson Correlation (Numerical Features)")
plt.show()

"""
Here, we see the overall correlation plot. New columns are visible, among which 'is_elderly' and 'has_risk_factors' stand out
with relatively high correlations with 'stroke'. (Of course, I'm mainly considering this in relation to the correlation values 
with the 'stroke' column, and since the highest correlation here is, for example, 0.24, I assume that's relatively high ;P)

One might consider removing columns with very low correlations with 'stroke', or columns that have been derived from others 
(for instance, removing 'age' since 'is_elderly' was created from it), but for now, I won't do that because even though 
the values don't suggest high correlations, these columns might still prove significant when building models.)
"""
cor_mat = healthcare.corr().round(2)
plt.figure(figsize=(10,10))
sns.heatmap(cor_mat, vmin=-1, vmax=1, annot=True, cmap='coolwarm')
plt.title("Pearson Correlation (Numerical Features)")
plt.show()

"""
I also created histograms for all columns.
It's clear that the 'stroke' column is very imbalanced, so oversampling will be necessary.
"""

plt.figure(figsize=(15, 15))

for i in range(0,15):
    plt.subplot(8, 2, i + 1)  # 8 rows, 2 columns, i+1-th plot
    plt.hist(healthcare.iloc[:, i], color='purple', bins=20, edgecolor='black')
    plt.title(f"Histogram: {healthcare.columns[i]}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.subplots_adjust(hspace=0.8)
plt.show()

"""
Then, I checked the final number of observations and split the data into X and y, 
after which I performed oversampling to balance the classes in the column to be predicted.
I checked the number of all observations and the class distribution in the 'stroke' column after performing SMOTE.
It's noticeable that the classes were balanced, with many new observations created based on the existing ones.
"""

print("TOTAL NUMBER OF OBSERVATIONS (before SMOTE): ", len(healthcare))

X = healthcare.drop('stroke', axis=1)
y = healthcare['stroke']

print("Distribution of 'stroke' column after SMOTE:\n", y.value_counts())

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("TOTAL NUMBER OF OBSERVATIONS (after SMOTE): ", len(y_resampled))
print("Distribution of 'stroke' column after SMOTE:\n", y_resampled.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, stratify=y_resampled, test_size=0.2, random_state=42)

"""
First, I created a random forest model and checked the metrics using the classification_report.
I also separately calculated Cohen's Kappa.
I displayed the confusion matrix as well.
The majority of the cases were correctly classified, and the metrics seem fairly solid, 
but it’s still not the perfect model. 95 observations were misclassified.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))
print(f"Cohen Kappa: {cohen_kappa_score(y_test, y_pred_rf)}")

# confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['No Stroke', 'Stroke'])
cm_disp_rf.plot(cmap='Blues', values_format='d')

plt.title("Confusion Matrix - Random Forest")
plt.show()

# Displaying the feature importance for this model. One might consider removing the least important features.
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind='bar', figsize=(15,5), title='Feature Importance - Random Forest')

"""
Similarly to the random forest, I created an XGBoost model. I checked its metrics and the confusion matrix.
The results are very close to the previous model, and even a bit better. Here, 90 observations were misclassified, 
which is 5 fewer than before.
"""

from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("XGBoost Report:")
print(classification_report(y_test, y_pred_xgb))
print(f"Cohen Kappa: {cohen_kappa_score(y_test, y_pred_xgb)}")

# confusion matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
cm_disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=['No Stroke', 'Stroke'])
cm_disp_xgb.plot(cmap='Purples', values_format='d')

plt.title("Confusion Matrix - XGBoost")
plt.show()

# Displaying the feature importance for this model
importances_xgb = pd.Series(xgb.feature_importances_, index=X.columns)
importances_xgb.sort_values(ascending=False).plot(kind='bar', figsize=(15,5), title='Feature Importance - XGBoost')


"""
Finally, I created a simple neural network to compare its performance with the previous models. 
However, it ultimately achieves slightly worse metrics.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_nn = scaler.fit_transform(X_train)
X_test_nn = scaler.transform(X_test)

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # binary classification, so sigmoid and 1 output layer

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_nn, y_train, epochs=30, batch_size=32, validation_split=0.2)

# Model evaluation
loss, accuracy = model.evaluate(X_test_nn, y_test)
print(f"Neural Network Accuracy: {accuracy:.4f}")

"""
As before, I calculated the Cohen Kappa coefficient and the confusion matrix. 
As you can see, this model performs the worst among those presented.
"""

y_pred_nn = (model.predict(X_test_nn) > 0.5).astype("int32")  # binary classification, so we convert probability to 0 or 1
print(f"Cohen Kappa: {cohen_kappa_score(y_test, y_pred_nn)}")

# Confusion matrix
cm_nn = confusion_matrix(y_test, y_pred_nn)
cm_disp_nn = ConfusionMatrixDisplay(confusion_matrix=cm_nn, display_labels=['No Stroke', 'Stroke'])
cm_disp_nn.plot(cmap='Blues', values_format='d')

plt.title("Confusion Matrix - Neural Network")
plt.show()


"""
To summarize, I managed to create predictive models that show quite high metric results, 
although they are not perfect. There are still many observations that were incorrectly classified. 
Hyperparameter tuning could be a next step, but for now, this is what I was able to achieve. 
I also tried to visualize the data in some way, although I know I didn’t include too many plots in the end, 
so that’s something to work on. I’m wondering what else I could do with this data or if there’s anything 
I did not do quite right – I’d be happy to receive suggestions or constructive criticism ;P
"""