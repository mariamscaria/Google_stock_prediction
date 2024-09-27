# Google_stock_prediction
This repository demonstrates the use of TensorFlow to predict Google's stock prices. It starts by gathering historical stock data from Yahoo Finance, followed by a thorough data cleaning and preparation process to ensure high-quality inputs for the model.  The dataset is then split into training and testing sets. 

    import pandas as pd
    stock_data = pd.read_csv('/Users/Downloads/GOOG.csv')

    stock_data.head()
    df = pd.read_csv('/Users/Downloads/GOOG.csv')

    sum_of_null = df.isnull().sum()
    duplicates = df.duplicated().sum()
    datatype = stock_data.dtypes

    stock_data['date'] = pd.to_datetime(stock_data['date']).dt.date
    df = stock_data.set_index('date').sort_index()
    df.index = pd.to_datetime(df.index)

    df.drop(columns=['symbol', 'adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor'], inplace=True)

    import matplotlib.pyplot as plt
    close = df.iloc[:, 2:3].values  # Assuming the closing price column is at index 2

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, closing_prices, color='red')
    plt.title('The price closing values')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.grid(True)
    plt.show()

    from sklearn.preprocessing import MinMaxScaler

    if df.shape[1] >= 3:

    close = df.iloc[:, 2].values  # Assuming closing prices are in the third column

    # Reshape the closing prices to fit the scaler
    closereshaped = close.reshape(-1, 1)

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the data with the scaler
    data_normalized = scaler.fit_transform(closereshaped)

    # Optionally, convert the normalized data back to a DataFrame for further use
    df['normalized_close'] = data_normalized

    # Inspect the normalized data
    df[['normalized_close']].head()
    else:
     "DataFrame does not have enough columns to normalize closing prices."

    from sklearn.model_selection import train_test_split
    
Ensure that data_normalized exists and is a 2D array
  
    if 'data_normalized' in locals():
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data_normalized, test_size=0.2, shuffle=False)
    "Training data shape: {train_data.shape}"
    "Testing data shape: {test_data.shape}"
    else:
    "data_normalized is not defined. Ensure that the normalization step is completed successfully."


    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from keras.preprocessing.sequence import TimeseriesGenerator
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

Assuming df is already defined and has a 'close' column
     
    close = df['close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(close)

Split data into training and testing sets

    train_size = int(len(data_normalized) * 0.8)
    train_data, test_data = data_normalized[:train_size], data_normalized[train_size:]

Define time step

    time_step = 100

Prepare data function

    def prepare_data(data, time_step):
    generator = TimeseriesGenerator(data, data, length=time_step, batch_size=1)
    return generator

Prepare generators for training and testing data

    train_generator = prepare_data(train_data, time_step)
    test_generator = prepare_data(test_data, time_step)

Build the LSTM model

    model = Sequential()
    model.add(LSTM(units=25, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

Train the model

    model.fit(train_generator, epochs=3)

Generate predictions

    train_predictions = model.predict(train_generator)
    test_predictions = model.predict(test_generator)

Rescale the data back to the original scale

    train_predictions_rescaled = scaler.inverse_transform(train_predictions)
    test_predictions_rescaled = scaler.inverse_transform(test_predictions)
    train_actual_rescaled = scaler.inverse_transform(train_data[time_step:])
    test_actual_rescaled = scaler.inverse_transform(test_data[time_step:])

Evaluate model performance metrics
     
    mse_train = mean_squared_error(train_actual_rescaled, train_predictions_rescaled)
    mae_train = mean_absolute_error(train_actual_rescaled, train_predictions_rescaled)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(train_actual_rescaled, train_predictions_rescaled)

    mse_test = mean_squared_error(test_actual_rescaled, test_predictions_rescaled)
    mae_test = mean_absolute_error(test_actual_rescaled, test_predictions_rescaled)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(test_actual_rescaled, test_predictions_rescaled)

Create a DataFrame to store the metrics

    results = pd.DataFrame({
    'Metric': ['MSE', 'MAE', 'RMSE', 'R2'],
    'Train': [mse_train, mae_train, rmse_train, r2_train],
    'Test': [mse_test, mae_test, rmse_test, r2_test]
    })

Export the results to a CSV file
    
    results.to_csv('model_metrics.csv', index=False)

Verify if the file is created
    
    import os
    if os.path.exists('model_metrics.csv'):
    "The file 'model_metrics.csv' has been created successfully."
    else:
    "Failed to create the file 'model_metrics.csv'."


