import pickle
from flask import Flask, request, jsonify, render_template # jsonify means that I return my result in the form of JSON., render_template will be responsible in finding out the URL of the HTML file.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# first step is that obviously I have created my flask name.
application = Flask(__name__)
app = application


# Now the first step, you know, as I said, my web application should be able to interact with the ridge.pickle and standard.pickle, Because it will be able to interact and it will be able to do the transformation function and also transformation when I say it is nothing but feature scaling, in case of ridge, it will be able to do the prediction.
#### import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb')) #rb meand open the file in read byte mode


# create home page
@app.route("/")
# return render template and give something like index dot HTML as soon as we write render template It will just go and find index dot HTML where it is over here. And it will always make sure that it will find inside a folder which is called as templates folder. So render_template basically means it will just look for the template folder over there. And inside that whether that index dot HTML file is present or not. If it is not present, then when we are executing this code, it will not work. So here I will just create a file which is called as index dot HTML
def index():
    return render_template('index.html')


# for the prediction of the model, iam going to create another app.route with another url /predictdata
@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    #  whether it is Get or post. If we really need to find out then we have to write this particular condition. If the request dot method is double equal to post , then I will do some execution. So I'll keep pass over here  And I'm not writing what I'm going to do because whenever it is a post I need to interact with my ridge model. I need to do the prediction, get the output. This I will write the code later on.
    if request.method == "POST":
        #pass 
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)

        # I need to show this result  And this result will be a list of value which will have just one value and it will be in the form of list. So here I will just create a variable which will be results here this will get assigned to my result of zero
        return render_template('home.html',results=result[0])


    #in the else block if it is not post then it is get
    else:
        return render_template('home.html')

# Exeution of if else block on web
# if I want to call this this URL slash predict data. So I will just go and call. Now what will happen once I call. By default this becomes a getter method in get method. What is happening? See if the request dot method is post. But right now it is not post. We will go into this else block. It has to show this home dot HTML which has this particular form.These are all my fields. And right now the result is empty because we did not do any post right now. Main thing is that we'll put values over here. And once we click this predict button it will just go and hit this predict_data point which is this function. Now it will say that if this particular post then we have to write our code over here  Now this is the thing that we are going to handle. Well, the first thing over here that we are going to do is that read all the inputs of all this particular values like like temperature, RH, W, S, rain, FM, Cdmc, ISI classes and region. So for that I will just remove this pass,


# when we have given by default 0.0.0 as my host address, This is basically mapped to the local IP address of any machine that you are working. So let's say if you are working, if you are running this entire application in a local machine, then also if you give 0.0.0, in short, this is getting mapped to your local IP address.
if __name__ == "__main__":
    app.run(host="0.0.0.0")