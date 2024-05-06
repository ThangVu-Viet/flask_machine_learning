import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle


def read_data(file_path):
    df = pd.read_csv(file_path).drop("X1 transaction date", axis=1)
    x = df.drop("Y house price of unit area", axis=1)  # bỏ cột dự báo khỏi bảng dữ liệu: bảng đặc trưng
    y1 = df['Y house price of unit area']
    y = np.array(df['Y house price of unit area']).reshape(y1.shape[0], 1)  # cột dự báo
    return x, y


def preprocessing(x):
    # Define polynomial degree and create PolynomialFeatures transformer
    degree = 2  # đa thức bậc 2
    polynomial_converter = PolynomialFeatures(degree=degree, include_bias=False)  # khởi tạo mô hình y = ax^2+bx+c

    # Transform input features into polynomial features
    poly_features = polynomial_converter.fit_transform(x)  # đi tìm đường cong tốt nhất mà gần với bộ dữ liệu X nhất

    # Split the data into training and testing sets

    return poly_features


def create_model_regression(x, y):
    from sklearn.linear_model import LinearRegression  # import library
    poly_features = preprocessing(x)
    # test
    x_train, x_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
    polymodel = LinearRegression()  # model hồi quy
    polymodel.fit(x_train, y_train)
    y_pred = polymodel.predict(x_test)
    MAE_Poly = round(metrics.mean_absolute_error(y_test, y_pred), 2)
    MSE_Poly = round(metrics.mean_squared_error(y_test, y_pred), 2)
    metrics_rgr_df = [MAE_Poly, MSE_Poly]
    # save model to file
    pickle.dump(polymodel, open('static/polymodel.pkl', 'wb'))
    return metrics_rgr_df


def create_model_boosting(x, y):
    from sklearn.ensemble import GradientBoostingRegressor
    poly_features = preprocessing(x)  # call function to get feature
    x_train, x_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
    # Initialize and fit the Gradient Boosting model
    gb_model = GradientBoostingRegressor()
    gb_model.fit(x_train, y_train)
    # Predict on test data
    y_pred_gb = gb_model.predict(x_test)
    # Calculate metrics for measure performance of algorithm
    MAE_gb = round(metrics.mean_absolute_error(y_test, y_pred_gb), 2)
    MSE_gb = round(metrics.mean_squared_error(y_test, y_pred_gb), 2)

    # Display metrics
    metrics_gb_df = [MAE_gb, MSE_gb]
    # save model to file
    pickle.dump(gb_model, open('static/gb_model.pkl', 'wb'))
    return metrics_gb_df


def create_model_RNN(x, y):
    from tensorflow.keras.models import Sequential, save_model
    from tensorflow.keras.layers import Dense, LSTM
    import h5py
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

    # Define RNN model
    rnn_model = Sequential()
    rnn_model.add(LSTM(120, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    rnn_model.add(LSTM(64, return_sequences=False))
    rnn_model.add(Dense(25))
    rnn_model.add(Dense(1))
    # Compile model
    rnn_model.compile(optimizer='adam', loss='mse')

    # Fit model
    rnn_model.fit(x_train, y_train, epochs=25, batch_size=1)

    # Predict on test data
    y_pred_rnn = rnn_model.predict(x_test).flatten()

    # Calculate metrics
    MAE_rnn = round(metrics.mean_absolute_error(y_test, y_pred_rnn), 2)
    MSE_rnn = round(metrics.mean_squared_error(y_test, y_pred_rnn), 2)
    # Display metrics
    metrics_rnn_df = [MAE_rnn, MSE_rnn]
    # save model to file
    save_model(rnn_model, 'static/rnn_model.h5')
    return metrics_rnn_df


def training_rgr():
    file_path = 'static/Data_training.csv'
    x, y = read_data(file_path)
    metrics_rgr_df = create_model_regression(x, y)
    metrics_gb_df = create_model_boosting(x, y)
    metrics_rnn_df = create_model_RNN(x, y)
    print("Build Model Successfully")
    return metrics_rgr_df, metrics_gb_df, metrics_rnn_df
