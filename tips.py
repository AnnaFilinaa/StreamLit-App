import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns


path = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'
tips = pd.read_csv(path)
st.write("""
# Анализ чаевых ресторана
""")
         
st.write("""
Общая выручка ресторана
""")

fig,ax=plt.subplots()
sns.histplot(tips['total_bill'])
plt.xlabel('Сумма счета')
plt.ylabel('Количество')
st.pyplot(fig)

st.write("""
## Зависимость между суммой счета и размером чаевых
""")
         
fig,ax=plt.subplots()
sns.scatterplot(data=tips, x='total_bill', y='tip')
plt.xlabel('Сумма счета')
plt.ylabel('Чаевые')
st.pyplot(fig)

st.write("""
## Зависимости размера чаевых от суммы счета и количества гостей
""")

fig,ax=plt.subplots()
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='size')
plt.xlabel('Сумма счета')
plt.legend(title='Количество гостей')
plt.ylabel('Чаевые')
st.pyplot(fig)

st.write("""
## Распределение объема выручки ресторана по дням недели
""")
         
fig,ax=plt.subplots()
sns.boxplot(data=tips, x="day", y="total_bill")
plt.xlabel('День недели')
plt.ylabel('Сумма счета')
st.pyplot(fig)

st.write("""
Зависимость размера чаевых от дня недели и пола официанта
""")

fig,ax=plt.subplots()
sns.scatterplot(x='tip', y='day', hue='sex', data=tips)
plt.xlabel('Чаевые')
plt.ylabel('День недели')
st.pyplot(fig)

st.write("""
Распределения суммы счета по дням недели с учетом времени обслуживания
""")

fig,ax=plt.subplots()
sns.boxplot(data=tips, x='day', y='total_bill', hue='time')
plt.xlabel('День недели')
plt.ylabel('Сумма счета')
st.pyplot(fig)