from flask import render_template, request
from app.page import bp
from app.page.Build_Model_Regression import preprocessing, training_rgr, read_data
import pickle
from tensorflow.keras.models import load_model
import pandas as pd
from app.page.db_process import db_process
from app.models.result import Result


@bp.route('/', methods=['GET', 'POST'])
def index():
    return render_template("page/index.html")


@bp.route('/training', methods=['GET', 'POST'])
def training():
    if request.method == "POST":
        metrics = training_rgr()
        sms_rgr = "Training model regression complete!!!"
        sms_bst = "Training model gradient regression complete!!!"
        sms_rnn = "Training model recurrent neural network complete!!!"
        text_out = [sms_rgr, sms_bst, sms_rnn]
        return render_template("page/index.html", text_out=text_out, metrics=metrics)
    return render_template("page/index.html")


@bp.route('/testing', methods=['GET', 'POST'])
def testing():
    if request.method == "POST":
        x_test, y_test = read_data('static/Data_testing.csv')
        poly_features = preprocessing(x_test)

        model_rgr = pickle.load(open("static/polymodel.pkl", "rb"))
        y_pred_rgr = model_rgr.predict(poly_features)  # the result of the prediction by using regression

        model_bst = pickle.load(open("static/gb_model.pkl", "rb"))
        y_pred_bst = model_bst.predict(poly_features)

        model_rnn = load_model("static/rnn_model.h5")
        y_pred_rnn = model_rnn.predict(x_test).flatten()

        # save data to sql
        db_process(y_test, y_pred_rgr, y_pred_bst, y_pred_rnn)
        # message save data successfully
        mss = "# Save data successfully"
        return render_template("page/index.html", mss=mss)
    return render_template("page/index.html")


@bp.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == "POST":
        results = Result.query.all() # the query all data in the table
        return render_template("page/index.html", results=results)
    return render_template("page/index.html")
