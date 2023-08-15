from flask import Flask, render_template, request, abort, Response, redirect, url_for
import json
import random
from datetime import datetime
from dateutil.relativedelta import relativedelta
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy

app = Flask(__name__)

# Simple form handling using raw HTML forms
@app.route('/', methods=['GET', 'POST'])
def sign_up():
    error = ""
    if request.method == 'POST':
        # Form being submitted; grab data from form.
        data = {'DATE': request.form['DATE'],
                'Temperature1': request.form['Temperature1'],
                'Rainfall1': request.form['Rainfall1'],
                'Temperature2': request.form['Temperature2'],
                'Rainfall2': request.form['Rainfall2'],
                'Temperature3': request.form['Temperature3'],
                'Rainfall3': request.form['Rainfall3'],
                'number': request.form['number']}

        if request.form['plot'] == 'yes':
            return get_plots(data, request.form['Precepitation'])
        return get_weather(data)

    # Render the sign-up page
    return render_template('info.html', message=error)

@app.route('/forecast/plots')
def get_plots(data, precepetation):
    date = datetime.strptime(data['DATE'], '%Y-%m-%d').date()
    date1 = copy.copy(date)

    X_pred1 = np.array([[0] * 3 + [data['Temperature1'], 0, 0, data["Rainfall1"]] + [date1.day, date1.month, date1.year]]).reshape((1, 10))
    date1 = date1 + relativedelta(days=1)
    X_pred2 = np.array([[0] * 3 + [data['Temperature2'], 0, 0, data["Rainfall2"]] + [date1.day, date1.month, date1.year]]).reshape((1, 10))
    date1 = date1 + relativedelta(days=1)
    X_pred3 = np.array([[0] * 3 + [data['Temperature3'], 0, 0, data["Rainfall3"]] + [date1.day, date1.month, date1.year]]).reshape((1, 10))

    scaler = joblib.load('scaler.save')
    X_pred1 = scaler.transform(X_pred1)[0][[3, 6, 7, 8, 9]]
    X_pred2 = scaler.transform(X_pred2)[0][[3, 6, 7, 8, 9]]
    X_pred3 = scaler.transform(X_pred3)[0][[3, 6, 7, 8, 9]]

    X_pred = np.append([X_pred1], [X_pred2], axis=0)
    X_pred = np.append(X_pred, [X_pred3], axis=0)
    X_pred = X_pred.reshape(1, 3, 5)

    number = int(data['number'])
    if number > 7:
        number = 7

    new_model = tf.keras.models.load_model('model.h5')
    regressor = joblib.load('regressor.save')
    forcasted = []
    predicted_temp = []

    for i in range(number):
        if (i < 3):
            yhat = new_model.predict(X_pred)
            predicted_temp = list(yhat[0][i])
            predicted_temp = scaler.inverse_transform(np.array([0, 0, 0] + predicted_temp + [0, 0, 0, 1, 1, 2023]).reshape((1, 10)))[0][3:4]
        else:
            if (i == 3):
                X3 = np.append([X_pred2], [X_pred3], axis=0)
                X3 = np.append(X3, [pred_X1], axis=0)
                X3 = X3.reshape(1, 3, 5)
                yhat = new_model.predict(X3)
                predicted_temp = list(yhat[0][0])
                predicted_temp = scaler.inverse_transform(np.array([0, 0, 0] + predicted_temp + [0, 0, 0, 1, 1, 2023]).reshape((1, 10)))[0][3:4]
            elif (i == 4):
                X4 = np.append([X_pred3], [pred_X1], axis=0)
                X4 = np.append(X4, [pred_X2], axis=0)
                X4 = X4.reshape(1, 3, 5)
                yhat = new_model.predict(X4)
                predicted_temp = list(yhat[0][0])
                predicted_temp = scaler.inverse_transform(np.array([0, 0, 0] + predicted_temp + [0, 0, 0, 1, 1, 2023]).reshape((1, 10)))[0][3:4]
            elif (i == 5):
                X5 = np.append([pred_X1], [pred_X2], axis=0)
                X5 = np.append(X5, [pred_X3], axis=0)
                X5 = X5.reshape(1, 3, 5)
                yhat = new_model.predict(X5)
                predicted_temp = list(yhat[0][0])
                predicted_temp = scaler.inverse_transform(np.array([0, 0, 0] + predicted_temp + [0, 0, 0, 1, 1, 2023]).reshape((1, 10)))[0][3:4] - 4
            elif (i == 6):
                X6 = np.append([pred_X2], [pred_X3], axis=0)
                X6 = np.append(X6, [pred_X4], axis=0)
                X6 = X6.reshape(1, 3, 5)
                yhat = new_model.predict(X6)
                predicted_temp = list(yhat[0][0])
                predicted_temp = scaler.inverse_transform(np.array([0, 0, 0] + predicted_temp + [0, 0, 0, 1, 1, 2023]).reshape((1, 10)))[0][3:4] - 3.5

        date2 = date + relativedelta(days=i+3)
        x = np.array([float(predicted_temp), date2.day, date2.month, date2.year]).reshape((1, 4))
        y = regressor.predict(x)[0]
        res = [abs(round(k, 2)) for k in y]
        temp = [abs(round(k, 2)) for k in predicted_temp]
        forcasted.append([str(date + relativedelta(days=i+3))] + [res[0], res[1], res[2], temp[0], res[3], res[4], res[5]])

        if (i == 0):
            pred_X1 = np.array([temp[0], res[5], date2.day + i, date2.month, date2.year])
        elif (i == 1):
            pred_X2 = np.array([temp[0], res[5], date2.day, date2.month, date2.year])
        elif (i == 2):
            pred_X3 = np.array([temp[0], res[5], date2.day, date2.month, date2.year])
        elif (i == 3):
            pred_X4 = np.array([temp[0], res[5], date2.day, date2.month, date2.year])
        elif (i == 4):
            pred_X5 = np.array([temp[0], res[5], date2.day, date2.month, date2.year])
        elif (i == 5):
            pred_X6 = np.array([temp[0], res[5], date2.day, date2.month, date2.year])

    all = pd.read_csv("data.csv")
    i = 0
    for i in range(all.shape[0]):
        if (all["Date"][i] == str(date + relativedelta(days=3))):
            break
    actual = all[i:i+number]
    actual.set_index('Date', inplace=True)
    actual.drop(['day'], axis=1, inplace=True)
    actual.drop(['month'], axis=1, inplace=True)
    actual.drop(['year'], axis=1, inplace=True)

    t = np.array(forcasted)
    df = pd.DataFrame(t, columns=['Date', 'DIFFUSED RADIATION (W/M^2)', 'GLOBAL RADIATION (W/M^2)', 'RELATIVE HUMIDITY (%)',
                                  'TEMPERATURE (°C)', 'MAXIMUM TEMPERATURE (°C)', 'MINIMUM TEMPERATURE (°C)', 'RAINFALL (MM)'])
#    fig, ax = plt.subplots()
    for name in df.columns[1:]:
        df[name] = df[name].astype(float)
    
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    sns.set(style='dark')
    px = 1 / plt.rcParams['figure.dpi']
    plt.rcParams['figure.figsize'] = 700 * px, 300 * px

    if (int(precepetation) != 1):
        if (int(precepetation) != 2):
            ax = sns.lineplot(data=df['TEMPERATURE (°C)'], markers=True, dashes=False, label='Forecasted Temperature')
            ax.spines['bottom'].set_color('white')

            ax.tick_params(colors='white', which='both')
            ax.yaxis.label.set_color('white')
            plt.savefig("static/forecasted_temp_plot.png", transparent=True, dpi=300)
            plt.close()

        ax1 = sns.lineplot(data=df['TEMPERATURE (°C)'], markers=True, dashes=False, label='Forecasted Temperature')
        ax1 = sns.lineplot(data=actual['Temperature (°C)'], markers=True, dashes=False, label='Actual Temperature')
        ax1.spines['bottom'].set_color('white')

        ax1.tick_params(colors='white', which='both')
        ax1.yaxis.label.set_color('white')
        plt.savefig("static/compared_temp_plot.png", transparent=True, dpi=300)
        plt.close()

    if (int(precepetation) != 0):
        if (int(precepetation) != 2):
            ax2 = sns.lineplot(data=df['RAINFALL (MM)'], markers=True, dashes=False, label='Forecasted Rainfall')
            ax2.spines['bottom'].set_color('white')

            ax2.tick_params(colors='white', which='both')
            ax2.yaxis.label.set_color('white')
            plt.savefig("static/forecasted_rainfall_plot.png", transparent=True, dpi=300)
            plt.close()

        ax3 = sns.lineplot(data=df['RAINFALL (MM)'], markers=True, dashes=False, label='Forecasted Rainfall')
        ax3 = sns.lineplot(data=actual['Rainfall (mm)'], markers=True, dashes=False, label='Actual Rainfall')

        ax3.spines['bottom'].set_color('white')

        ax3.tick_params(colors='white', which='both')
        ax3.yaxis.label.set_color('white')
        plt.savefig("static/compared_rainfall_plot.png", transparent=True, dpi=300)
        plt.close()

    if (int(precepetation) == 2):
        ax4 = sns.lineplot(data=df['MAXIMUM TEMPERATURE (°C)'], markers=True, dashes=False, label='Forecasted Maximum Temperature')
        ax4 = sns.lineplot(data=actual['Maximum temperature (°C)'], markers=True, dashes=False, label='Actual Maximum Temperature')
        ax4.spines['bottom'].set_color('white')

        ax4.tick_params(colors='white', which='both')
        ax4.yaxis.label.set_color('white')
        plt.savefig("static/compared_max_temp_plot.png", transparent=True, dpi=300)
        plt.close()

        ax5 = sns.lineplot(data=df['MINIMUM TEMPERATURE (°C)'], markers=True, dashes=False, label='Forecasted Minimum Temperature')
        ax5 = sns.lineplot(data=actual['Minimum temperature (°C)'], markers=True, dashes=False, label='Actual Minimum Temperature')

        ax5.spines['bottom'].set_color('white')

        ax5.tick_params(colors='white', which='both')
        ax5.yaxis.label.set_color('white')
        plt.savefig("static/compared_min_temp_plot.png", transparent=True, dpi=300)
        plt.close()

    return render_template('plot.html', title='Weather App', data=[precepetation])

@app.route('/forecast')
def get_weather(data):
    date = datetime.strptime(data['DATE'], '%Y-%m-%d').date()
    date1 = copy.copy(date)

    X_pred1 = np.array([[0] * 3 + [data['Temperature1'], 0, 0, data["Rainfall1"]] + [date1.day, date1.month, date1.year]]).reshape((1, 10))
    date1 = date1 + relativedelta(days=1)
    X_pred2 = np.array([[0] * 3 + [data['Temperature2'], 0, 0, data["Rainfall2"]] + [date1.day, date1.month, date1.year]]).reshape((1, 10))
    date1 = date1 + relativedelta(days=1)
    X_pred3 = np.array([[0] * 3 + [data['Temperature3'], 0, 0, data["Rainfall3"]] + [date1.day, date1.month, date1.year]]).reshape((1, 10))

    scaler = joblib.load('scaler.save')
    X_pred1 = scaler.transform(X_pred1)[0][[3, 6, 7, 8, 9]]
    X_pred2 = scaler.transform(X_pred2)[0][[3, 6, 7, 8, 9]]
    X_pred3 = scaler.transform(X_pred3)[0][[3, 6, 7, 8, 9]]
   
    X_pred = np.append([X_pred1], [X_pred2], axis=0)
    X_pred = np.append(X_pred, [X_pred3], axis=0)
    X_pred = X_pred.reshape(1, 3, 5)

    number = int(data['number'])
    if number > 7:
        number = 7

    new_model = tf.keras.models.load_model('model.h5')
    regressor = joblib.load('regressor.save')
    forcasted = []
    predicted_temp = []

    for i in range(number):
        if (i < 3):
            yhat = new_model.predict(X_pred)
            predicted_temp = list(yhat[0][i])
            predicted_temp = scaler.inverse_transform(np.array([0, 0, 0] + predicted_temp + [0, 0, 0, 1, 1, 2023]).reshape((1, 10)))[0][3:4]
        else:
            if (i == 3):
                X3 = np.append([X_pred2], [X_pred3], axis=0)
                X3 = np.append(X3, [pred_X1], axis=0)
                X3 = X3.reshape(1, 3, 5)
                yhat = new_model.predict(X3)
                predicted_temp = list(yhat[0][0])
                predicted_temp = scaler.inverse_transform(np.array([0, 0, 0] + predicted_temp + [0, 0, 0, 1, 1, 2023]).reshape((1, 10)))[0][3:4]
            elif (i == 4):
                X4 = np.append([X_pred3], [pred_X1], axis=0)
                X4 = np.append(X4, [pred_X2], axis=0)
                X4 = X4.reshape(1, 3, 5)
                yhat = new_model.predict(X4)
                predicted_temp = list(yhat[0][0])
                predicted_temp = scaler.inverse_transform(np.array([0, 0, 0] + predicted_temp + [0, 0, 0, 1, 1, 2023]).reshape((1, 10)))[0][3:4]
            elif (i == 5):
                X5 = np.append([pred_X1], [pred_X2], axis=0)
                X5 = np.append(X5, [pred_X3], axis=0)
                X5 = X5.reshape(1, 3, 5)
                yhat = new_model.predict(X5)
                predicted_temp = list(yhat[0][0])
                predicted_temp = scaler.inverse_transform(np.array([0, 0, 0] + predicted_temp + [0, 0, 0, 1, 1, 2023]).reshape((1, 10)))[0][3:4] - 4
            elif (i == 6):
                X6 = np.append([pred_X2], [pred_X3], axis=0)
                X6 = np.append(X6, [pred_X4], axis=0)
                X6 = X6.reshape(1, 3, 5)
                yhat = new_model.predict(X6)
                predicted_temp = list(yhat[0][0])
                predicted_temp = scaler.inverse_transform(np.array([0, 0, 0] + predicted_temp + [0, 0, 0, 1, 1, 2023]).reshape((1, 10)))[0][3:4] - 3.5

        date2 = date + relativedelta(days=i+3)
        x = np.array([float(predicted_temp), date2.day, date2.month, date2.year]).reshape((1, 4))
        y = regressor.predict(x)[0]
        res = [abs(round(k, 2)) for k in y]
        temp = [abs(round(k, 2)) for k in predicted_temp]
        forcasted.append([str(date + relativedelta(days=i+3))] + [res[0], res[1], res[2], temp[0], res[3], res[4], res[5]])

        if (i == 0):
            pred_X1 = np.array([temp[0], res[5], date2.day + i, date2.month, date2.year])
        elif (i == 1):
            pred_X2 = np.array([temp[0], res[5], date2.day, date2.month, date2.year])
        elif (i == 2):
            pred_X3 = np.array([temp[0], res[5], date2.day, date2.month, date2.year])
        elif (i == 3):
            pred_X4 = np.array([temp[0], res[5], date2.day, date2.month, date2.year])
        elif (i == 4):
            pred_X5 = np.array([temp[0], res[5], date2.day, date2.month, date2.year])
        elif (i == 5):
            pred_X6 = np.array([temp[0], res[5], date2.day, date2.month, date2.year])

    return render_template('index.html', title='Weather App', data=forcasted)

if __name__ == "__main__":
    app.run()