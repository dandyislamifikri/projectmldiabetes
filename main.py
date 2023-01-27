import pandas as pd
import numpy as np
import streamlit as st

st.write("""
# Simple Diabetes Prediction App
This app predicts if a person has **Diabetes** or not!
""")

st.sidebar.header('User Input Parameters')


def user_input_features():
    num_preg = st.sidebar.slider('num_preg', 1.0, 10.0, 1.0)
    glucose_conc = st.sidebar.slider('glucose_conc', 44.0, 144.0, 44.0)
    diastolic_bp = st.sidebar.slider('diastolic_bp', 60.0, 110.0, 60.0)
    #thickness = st.sidebar.slider('thickness', 13.0, 50.0, 13.0)
    insulin = st.sidebar.slider('insulin', 100.0, 500.0, 100.0)
    bmi = st.sidebar.slider('bmi', 20.0, 47.0, 20.0)
    diab_pred = st.sidebar.slider('diab_pred', 0.1, 1.7, 0.1)
    age = st.sidebar.slider('age', 22.0, 65.0, 22.0)
    skin = st.sidebar.slider('skin', 0.9, 2.0, 0.9)
    data = {'num_preg': num_preg,
            'glucose_conc': glucose_conc,
            'diastolic_bp': diastolic_bp,
            'insulin': insulin,
            'bmi': bmi,
            'diab_pred': diab_pred,
            'age': age,
            'skin' : skin
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

data = pd.read_csv("pima-data.csv")

diabetes_map = {True: 1, False: 0}
data['diabetes'] = data['diabetes'].map(diabetes_map)

from sklearn.model_selection import train_test_split
feature_columns = ['num_preg', 'glucose_conc', 'diastolic_bp', 'insulin', 'bmi', 'diab_pred', 'age', 'skin']
predicted_class = ['diabetes']

X = data[feature_columns].values
y = data[predicted_class].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
random_forest_model.fit(X_train, y_train.ravel())

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier(n_estimators = 800, min_samples_split = 2,min_samples_leaf = 4, max_features = 'sqrt', max_depth = 90,bootstrap =True)
rf.fit(X_train,y_train.ravel())
#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
#rf_random.fit(X_train,y_train.ravel())

predict_train_data = random_forest_model.predict(X_test)
predict_train_data2 = rf.predict(X_test)

app_pred = rf.predict(df)

if app_pred == 0:
    st.write("This person does not have diabetes")
if app_pred == 1:
    st.write("This person has diabetes")

#from sklearn import metrics
#print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data2)))
#print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data2)))

#print('Best Hyperparameters: %s' % rf_random.best_params_)
