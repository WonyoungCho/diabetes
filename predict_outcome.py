import pickle
import streamlit as st
from urllib.request import urlopen
import time
import joblib

st.set_page_config(layout="wide")

def prediction(X_test):
    mfile = 'https://github.com/WonyoungCho/diabetes/raw/main/finalized_model.pkl'
    mfile = 'finalized_model.pkl'
    model = joblib.load(mfile)
    #with open(mfile, 'rb') as f:
    #    model = pickle.load(f)
    #model = pickle.load(urlopen(mfile))

    result = model.predict_proba([X_test])
    
    print(X_test)
    print(result)
    return result[0][1]

def input_values():
    cols = {'preg':'Number of times pregnant',
            'plas':'Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
            'pres':'Diastolic blood pressure (mm Hg)',
            'skin':'Triceps skin fold thickness (mm)',
            'test':'2-Hour serum insulin (mu U/ml)',
            'mass':'Body mass index (weight in kg/(height in m)^2)',
            'pedi':'Diabetes pedigree function',
            'age':'Age (years)'
            }
    
    keys = list(cols.keys())

    age  = st.slider(cols[keys[7]],  0,  100,  20)
    mass = st.slider(cols[keys[5]],  0,  200,  60)
    preg = st.slider(cols[keys[0]],  0,   10,   1)
    plas = st.slider(cols[keys[1]], 50,  200, 100)
    pres = st.slider(cols[keys[2]], 30,  200,  80)
    skin = st.slider(cols[keys[3]], 30,  200,  80)
    test = st.slider(cols[keys[4]],  0, 1000,  80)
    pedi = st.slider(cols[keys[6]],0.0,  3.0, 1.1)

    #X_test = np.array([preg,plas,pres,skin,test,mass,pedi,age])
    X_test = [preg,plas,pres,skin,test,mass,pedi,age]
    result = prediction(X_test)

    return result

def main():
    result = input_values()    
    
    with st.sidebar:
        # st.balloons()
        st.markdown(f'# Probability for diabetes')
        st.markdown(f'# {result*100:.2f} %')
        now = time

        print(now.strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == '__main__':
    main()
