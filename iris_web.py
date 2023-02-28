#importar las librerias necesarias
import streamlit as st
import pickle
import pandas as pd

#extraer los archivos pickle
with open('lin_reg.pkl', 'rb') as li:
    lin_reg = pickle.load(li)
    
#with open('log_reg.pkl', 'rb') as lo:
#    log_reg = pickle.load(lo)
    
with open('svc_m.pkl', 'rb') as sv:
    svc_m = pickle.load(sv)
    
#funcion para clasificar las flores
def classify(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Versicolor'
    else:
        return 'Virginica'
    
def main():
    st.title('Modelamiento de Iris')
    st.sidebar.header('User Input Parameters')
    
    def user_input_parameters():
        sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'sepal_legth': sepal_length,
                'sepal_width': sepal_width,
                'petal_legth': petal_length,
                'petal_width': petal_width,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    
    df = user_input_parameters()
    
    option = ['SVM', 'Linear Regression',]
    model = st.sidebar.selectbox('Choose an option', option)
    
    st.subheader('User Input Parameters')
    st.subheader(model)
    st.write(df)
    
    if st.button('Run'):
        if model == 'SVM':
            st.success(classify(svc_m.predict(df)))
        #elif model == 'Logistic Regression':
        #    st.success(classify(log_reg.predict(df)))
        else:
            st.success(classify(lin_reg.predict(df)))

if __name__ == '__main__':
    main()