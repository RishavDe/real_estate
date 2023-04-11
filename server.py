# from flask import Flask, request, jsonify
import util
import streamlit as st


# @app.route('/get_locations', methods = ['GET'])
def get_location_names():
    response = jsonify({
        'locations' : util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# @app.route('/predict_price', methods=['GET', 'POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = request.form['BHK']

    response = jsonify({
        'estimated_price' : util.get_estimated_price(location, total_sqft, bhk)
    })

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response



if __name__ =="__main__":
    #print("Starting Python Flask Server For Home Price predictor")
    app.run(host='0.0.0.0', port=8000)
