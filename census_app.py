import streamlit as st
from sklearn import datasets
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from plotnine import *

st.title("ML application!")

st.write("""
# Explore the data
### Comparing classifiers""")

# st.selectbox("Choose data:", ("Iris", "Breast Cancer"))

# Make selectbox a sidebar
dataset_name = st.sidebar.selectbox("Choose data:", ("Census1", "Census2"))


## Classifiers
c_name = st.sidebar.selectbox("Choose an algorithm:", ("Logistic Regression", "SVM", "Linear SVM"))

def get_dataset(dataset_name):
     if dataset_name == "Census1":
         data = pd.read_csv("census.csv")
     else:
         data = pd.read_csv("census.csv")
     X = data[['graduate_attainment_pct','health_insurance_pct','below_poverty_lvl_pct','occupied_houses_pct','no_vehicles_hh_pct','pop_over_18_pct']]
     y = data['above_US_median_income']
     return X, y

X, y = get_dataset(dataset_name)
st.write("Obs:", X.shape[0])
st.write("Features:", X.shape[1])
st.write("Number of classes", len(np.unique(y)))

#### Add parameters
def add_param(c_name):
    params = dict()
    if c_name == "Logistic Regression":
        C = st.sidebar.slider("C: less regularization ---->", 0.001, 10.0)
        params["C"] = C
    elif c_name == "SVM":
        C = st.sidebar.slider("C: less regularization ---->", 0.001, 10.0)
        params["C"] = C
    else:
        C = st.sidebar.slider("C: less regularization ---->", 0.001, 10.0)
        params["C"] = C
    return params

#### Call add_param function with classifier name as argument (selected above)
params = add_param(c_name)

#### Add classifier: takes name of classifier we want to use and parameters for the one

def add_class(c_name, params):
    if c_name == "Logistic Regression":
        mod = LogisticRegression(C=params["C"])
    elif c_name == "SVM":
        mod = SVC(C=params["C"])
    else:
        mod = LinearSVC(C=params["C"])
    return mod

#### Call add class function to add classifier and associated parameters
mod = add_class(c_name, params)


### Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

mod.fit(X_train, y_train)
y_pred = mod.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Algorithm = {c_name}")
st.write(f"Accuracy (same as test) = {round(acc, 2)}")
st.write(f"Training score: {round(mod.score(X_train, y_train), 2)}")
st.write(f"Test score: {round(mod.score(X_test, y_test), 2)}")


#### Plot
import pandas as pd
import altair as alt


### Apply PCA to get 2 dimensions
pca = PCA(2)
X_plot = pd.DataFrame(pca.fit_transform(X))

X_plot.columns = ["x", "y"]

alt_plot = alt.Chart(X_plot).mark_circle().encode(x="x",
y="y").interactive()
alt_plot

### GIS
df = pd.read_excel("fips.xlsx", dtype={"fips": str})


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px
import plotly.graph_objects as go

fig = px.choropleth(df, geojson=counties, locations='fips', color='median_hh_income', hover_data=['text'],
                           color_continuous_scale="Viridis",
                           range_color=(0, 150000),
                           scope="usa",
                           labels={'median_hh_income':'Median Household Income'}
                          )


fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig)


## Another Version

#import plotly.graph_objects as go

#import pandas as pd
#df = pd.read_excel('fips.xlsx')

#for col in df.columns:
    #df[col] = df[col].astype(str)

#df['text'] = df['county_state_name'] + '<br>' + \
    #'% of Population with Graduate Degrees' + df['graduate_attainment_pct'] + '<br>' + \
    #'% of Population with Health Insurance' + df['health_insurance_pct'] + '<br>' + \
    #'% of Population Below Poverty Level' + df['below_poverty_lvl_pct'] + '<br>' + \
    #'% of Population Over 18 Years of Age' + df['pop_over_18_pct'] + '<br>' + \
    #'% of Houses Occupied' + df['occupied_houses_pct'] + '<br>' + \
    #'% of Households without a Vehicle' + df['no_vehicles_hh_pct']

#fig = go.Figure(data=go.Choropleth(
    #locations=df['fips'],
    #z=df['median_hh_income'].astype(float),
    #locationmode='geojson-id',
    #colorscale='Viridis',
    #autocolorscale=False,
    #range_color=(0, 150000),
    #text=df['text'], # hover text
    #marker_line_color='white', # line markers between states
    #colorbar_title="Median Household Income"
#))

#fig.update_layout(
    #title_text='US Counties by Median Household Income and Predictors',
    #geo = dict(
        #scope='usa',
        #projection=go.layout.geo.Projection(type = 'albers usa'),
        #showlakes=True, # lakes
        #lakecolor='rgb(255, 255, 255)'),
#)

#st.plotly_chart(fig)


## Another Attempt
