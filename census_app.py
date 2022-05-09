import streamlit as st
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from plotnine import *


st.write("""
## Median Household Incomes of US Counties""")

st.write("""
##### This project aims to predict whether the median household income of a given county in the USA is above or below the national median income""")

st.write("""
###### Use this app for visualizing the data, looking at a map of counties with detailed information and predicting whether a county is above or below the national median household income:""")


vizdf = pd.read_csv('fips.csv')

# Subset the data for Visualizations

vizdf['median_hh_income'] = vizdf['median_hh_income'].astype('float')
vizdf['graduate_attainment_pct'] = vizdf['graduate_attainment_pct'].astype('float')
vizdf['below_poverty_lvl_pct'] = vizdf['below_poverty_lvl_pct'].astype('float')

income_high = vizdf.nlargest(20, ['median_hh_income'])
income_low = vizdf.nsmallest(20, ['median_hh_income'])
graduate_high = vizdf.nlargest(20, ['graduate_attainment_pct'])
graduate_low = vizdf.nsmallest(20, ['graduate_attainment_pct'])
poverty_high = vizdf.nlargest(20, ['below_poverty_lvl_pct'])
poverty_low = vizdf.nsmallest(20, ['below_poverty_lvl_pct'])

##############
def main():
    page = st.selectbox(
            "Select an action",
            [
                "Exploratory Data Analysis", #First Page
                "Choropleth" #Second Page
            ]
    )
    if page == "Exploratory Data Analysis":
        eda()

    elif page == "Choropleth":
        choropleth()

############

def eda():
    st.header("Visualizations")
    sd = st.radio(
        "Select data to visualize:", #Drop Down Menu Name
        [
            "Top 20 Counties by Median Household Income", #First option
            "Bottom 20 Counties by Median Household Income", #Second option
            "Top 20 Counties by % of Population with Graduate Degrees", #Third option
            "Bottom 20 Counties by % of Population with Graduate Degrees",
            "Top 20 Counties by % of Population below Poverty Level", #Fourth option
            "Bottom 20 Counties by % of Population below Poverty Level"
        ]
    )

    fig = plt.figure(figsize=(12, 12))
    sns.set(style="darkgrid")

    if sd == "Top 20 Counties by Median Household Income":
        sns.barplot(y="county_state_name",  x="median_hh_income", data=income_high, color='darkblue')
        plt.title('Top 20 Counties by Median Household Income')

    elif sd == "Bottom 20 Counties by Median Household Income":
        sns.barplot(y="county_state_name",  x="median_hh_income", data=income_low, color='darkblue')
        plt.title('Bottom 20 Counties by Median Household Income')

    elif sd == "Top 20 Counties by % of Population with Graduate Degrees":
        sns.barplot(y="county_state_name",  x="graduate_attainment_pct", data=graduate_high, color='darkblue')
        plt.title('Top 20 Counties by % Population of Graduates')

    elif sd == "Bottom 20 Counties by % of Population with Graduate Degrees":
        sns.barplot(y="county_state_name",  x="graduate_attainment_pct", data=graduate_low, color='darkblue')
        plt.title('Bottom 20 Counties by % Population of Graduates')

    elif sd == "Top 20 Counties by % of Population below Poverty Level":
        sns.barplot(y="county_state_name",  x="below_poverty_lvl_pct", data=poverty_high, color='darkblue')
        plt.title('Top 20 Counties by % Population Below Poverty Level')

    elif sd == "Bottom 20 Counties by % of Population below Poverty Level":
        sns.barplot(y="county_state_name",  x="below_poverty_lvl_pct", data=poverty_low, color='darkblue')
        plt.title('Bottom 20 Counties by % Population Below Poverty Level')

    st.pyplot(fig)




############
# choropleth
df = pd.read_excel("fips.xlsx", dtype={"fips": str})

def choropleth():
    st.header("Map")

    from urllib.request import urlopen
    import json
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
            counties = json.load(response)

    fig = px.choropleth(df, geojson=counties, locations='fips', color='median_hh_income', hover_data=['text'],
                           color_continuous_scale="Viridis",
                           range_color=(0, 150000),
                           scope="usa",
                           labels={'median_hh_income':'Median Hh Income'}
                          )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    st.plotly_chart(fig)

if __name__ == "__main__":
    main()

##################


## Classifiers
c_name = st.selectbox("Choose an algorithm:", ("Logistic Regression", "SVM", "Linear SVM"))

data = pd.read_csv("census.csv")
X = data[['graduate_attainment_pct','health_insurance_pct','below_poverty_lvl_pct','occupied_houses_pct','no_vehicles_hh_pct','pop_over_18_pct']]
y = data['above_US_median_income']


st.write("Obs:", X.shape[0])
st.write("Features:", X.shape[1])
st.write("Number of classes", len(np.unique(y)))

#### Add parameters
def add_param(c_name):
    params = dict()
    if c_name == "Logistic Regression":
        C = st.slider("C: less regularization ---->", 0.001, 10.0)
        params["C"] = C
    elif c_name == "SVM":
        C = st.slider("C: less regularization ---->", 0.001, 10.0)
        params["C"] = C
    else:
        C = st.slider("C: less regularization ---->", 0.001, 10.0)
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


### Apply PCA Analysis for top Principal Components

##################
pcadata = pd.read_csv('fips.csv')

features = pcadata[['graduate_attainment_pct','health_insurance_pct','below_poverty_lvl_pct','occupied_houses_pct','no_vehicles_hh_pct','pop_over_18_pct']]


n_components = 4

pca = PCA(n_components=n_components)
components = pca.fit_transform(features)

total_var = pca.explained_variance_ratio_.sum() * 100

labels = {str(i): f"PC {i+1}" for i in range(n_components)}
labels['color'] = 'Median Hh Income'

fig = px.scatter_matrix(
    components,
    color=pcadata.median_hh_income,
    dimensions=range(n_components),
    labels=labels,
    title=f'Total Explained Variance: {total_var:.2f}%'
)

fig.update_traces(diagonal_visible=False)
st.plotly_chart(fig)

########### fin ###########
