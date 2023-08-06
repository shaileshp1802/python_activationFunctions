import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd

st.set_page_config(
    page_title="Activation Function by Shailesh Paliwal"
)


st.header("Plots of Different Activation Functions:")

number = st.number_input('Insert x for Range of Dataset (-x,x):', value=10)

x = np.linspace(-number, number)
fig, ax = plt.subplots()


options = st.multiselect(
    'Choose Activation Functions for details: ',
    ['Linear Activation Function', 'TanhX', 'binarySigmoid', 'softMax', 'ReLU'], ['Linear Activation Function'])

######## FUNCTION DEFINITIONS #######

def linearActivation(x):
    return x

def TanhX(x):
    return np.tanh(x)

def binarySigmoid(x):
    return 1/(1 + (np.exp(-x)))

def softMax(x):
    return (np.exp(x)/sum(np.exp(x)))

def ReLU(x):
    return np.maximum(0,x)


for option in options:
    if (option == 'Linear Activation Function'):
        ax.plot(x, linearActivation(x))

    if (option == 'TanhX'):
        ax.plot(x, TanhX(x))

    if (option == 'binarySigmoid'):
        ax.plot(x, binarySigmoid(x))

    if (option == 'softMax'):
        ax.plot(x, softMax(x))

    if (option == 'ReLU'):
        ax.plot(x, ReLU(x))

st.pyplot(fig)