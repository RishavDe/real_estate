import pickle
import json
import numpy as np
import streamlit as st

__locations = None
__data_columns = None
__model = None
__result = None
def get_estimated_price(location,sqft,bhk):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x =np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0])




def load_Saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    with open('columns.json', 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]



    with open('Kolkata_home_prices_predictor.pickle', 'rb') as f:
        __model = pickle.load(f)
    print("loading artifacts...done")

def get_location_names():
    return __locations

def main():
    global __result
    #st.beta_set_page_config(page_title='RealEstate_Price', layout='wide', initial_sidebar_state='auto')
    st.title("Kolkata Real Estate Price Predictor")

    location = st.selectbox("Location", __locations)
    bhk = st.text_input("BHK")
    area = st.text_input("Total Sq Feet")

    if(st.button("Predict")):
        __result = get_estimated_price(location, area, bhk)
        result = str(__result)
        if(len(result) <=6 ):
            st.write('Predicted Price:',result[0:2],' lacs (approx)')
        else:
            st.write('Predicted Price:', result[0:1], ' crores (approx)')

    # st.success('Predicted Property Price'.format(__result))




if __name__ == '__main__':
    load_Saved_artifacts()

    # print(get_location_names())
    # get_location_names()
    # print(get_estimated_price('Baghajatin', 1600, 2))
    # print(get_estimated_price('Joka', 1400, 3))
    # print(get_estimated_price('Barasat', 1700, 3))

    main()
    print("Model Successful")