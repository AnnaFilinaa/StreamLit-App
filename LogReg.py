import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class LogReg:
    def __init__(self, learning_rate, n_inputs):
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.coef_ = np.random.uniform(size=n_inputs)
        self.intercept_ = np.random.uniform()
        
    def fit(self, X, y):
        for _ in range(100000):
            y_pred = 1 / (1+np.exp(-(np.dot(X, self.coef_) + self.intercept_)))
            error = (y - y_pred)
            w0_grad =  np.mean(error, axis=0) 
            w_grad = np.mean(error * X.T, axis=1)

            self.coef_ += self.learning_rate * w_grad
            self.intercept_ += self.learning_rate * w0_grad
        return self.coef_, self.intercept_
    
    
    def predict(self, X):
        y_pred = 1 / (1+np.exp(-(np.dot(X, self.coef_) + self.intercept_)))
        return np.round(y_pred)


def scatterplot(x_column, y_column, data):
    fig, ax = plt.subplots()
    ax.scatter(data[x_column], data[y_column])
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    return fig


def barplot(x_column, y_column, data):
    fig, ax = plt.subplots()
    data.groupby(x_column)[y_column].mean().plot(kind='bar', ax=ax)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    return fig


def lineplot(x_column, y_column, data):
    fig, ax = plt.subplots()
    ax.plot(data[x_column], data[y_column])
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    return fig


st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Logistic Regression Demo')

uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file).drop('Unnamed: 0', axis=1)
    target_column = st.selectbox('Select the target column', data.columns)
    feature_columns = st.multiselect('Select feature columns', data.columns.tolist())
    X = data[feature_columns].values
    y = data[target_column].values

    model = LogReg(learning_rate=0.01, n_inputs=X.shape[1])
    coef, intercept = model.fit(X, y)
    weights = dict(zip(feature_columns, coef))

    st.write('Regression weights:')
    st.write(weights)
    
    plot_type = st.selectbox('Select plot type', ['Scatterplot', 'Bar plot', 'Line plot'])
    x_column = st.selectbox('Select x column', feature_columns)
    y_column = st.selectbox('Select y column', feature_columns)
    if plot_type == 'Scatterplot':
        fig = scatterplot(x_column, y_column, data)
    elif plot_type == 'Bar plot':
        fig = barplot(x_column, y_column, data)
    else:
        fig = lineplot(x_column, y_column, data)
    st.pyplot(fig)