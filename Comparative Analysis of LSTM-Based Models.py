#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Conv1D, MaxPooling1D, Flatten
from google.colab import files
uploaded = files.upload()



data = pd.read_excel('Yuanchang_imputed.xlsx', engine='openpyxl')
data['Date'] = pd.to_datetime(data['time'])
data = data[['Date', 'Yuanchang_2']].dropna()

end_date = pd.to_datetime('2022-12-31')

periods = {
    '1yr': end_date - pd.DateOffset(years=1) + pd.DateOffset(days=1),
    '5yr': end_date - pd.DateOffset(years=5) + pd.DateOffset(days=1),
    '10yr': end_date - pd.DateOffset(years=10) + pd.DateOffset(days=1),
    '20yr': end_date - pd.DateOffset(years=20) + pd.DateOffset(days=1),
}

data_sets = {
    'data_1yr': data[(data['Date'] >= periods['1yr']) & (data['Date'] <= end_date)].copy(),
    'data_5yr': data[(data['Date'] >= periods['5yr']) & (data['Date'] <= end_date)].copy(),
    'data_10yr': data[(data['Date'] >= periods['10yr']) & (data['Date'] <= end_date)].copy(),
    'data_20yr': data[(data['Date'] >= periods['20yr']) & (data['Date'] <= end_date)].copy(),
}

print("1 year data length:", len(data_sets['data_1yr']))
print("5 year data length:", len(data_sets['data_5yr']))
print("10 year data length:", len(data_sets['data_10yr']))
print("20 year data length:", len(data_sets['data_20yr']))





import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.plot(data['Date'], data['Yuanchang_2'], marker='o', linestyle='-', markersize=1)
plt.title('Yuanchang_2 Time Series')
plt.xlabel('Date')
plt.ylabel('level')
plt.grid(True)
plt.tight_layout()
plt.show()





def create_dataset(series, time_steps=10):
    X, y = [], []
    for i in range(len(series) - time_steps):
        X.append(series[i:i+time_steps])
        y.append(series[i+time_steps])
    return np.array(X), np.array(y)




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.regularizers import l2

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=input_shape,
                   kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))  # 防止 overfitting
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse')
    return model

def build_bilstm(input_shape):
    model = Sequential()
    model.add(Bidirectional(
        LSTM(32, activation='relu', kernel_regularizer=l2(0.001)),
        input_shape=input_shape
    ))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse')
    return model

def build_cnn_lstm(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu',
                     kernel_regularizer=l2(0.001), input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(LSTM(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse')
    return model

def build_cnn_bilstm(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu',
                     kernel_regularizer=l2(0.001), input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32, activation='relu', kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse')
    return model



from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score

results = []  

time_steps = 30
epochs = 64
batch_size = 32

for period_name, df in data_sets.items():
    values = df['Yuanchang_2'].values.reshape(-1, 1)
    dates = df['Date'].values 

    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    X, y = create_dataset(values_scaled, time_steps)

   
    dates = dates[time_steps:]

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    date_train, date_test = dates[:split], dates[split:]

    input_shape = (X_train.shape[1], X_train.shape[2])

    for model_name, build_fn in [('LSTM', build_lstm), ('BiLSTM', build_bilstm), ('CNN-LSTM', build_cnn_lstm),('CNN-BiLSTM', build_cnn_bilstm)]:
        model = build_fn(input_shape)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=epochs,
          batch_size=batch_size,
          verbose=0,
          callbacks=[early_stop])

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
          'Period': period_name,
          'Model': model_name,
          'MSE': mse,
          'RMSE': rmse,
          'R2': r2,
          'y_test': y_test.flatten(),
          'y_pred': y_pred.flatten(),
          'date_test': date_test
        })




import pandas as pd
results_df = pd.DataFrame(results)
results_df





results_df[['Period', 'Model', 'MSE', 'RMSE','R2']]





import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

# MSE
plt.figure(figsize=(12,6))
sns.barplot(data=results_df, x='Period', y='MSE', hue='Model')
plt.title('Model MSE Comparison by Time Period')
plt.ylabel('MSE')
plt.show()

# RMSE
plt.figure(figsize=(12,6))
sns.barplot(data=results_df, x='Period', y='RMSE', hue='Model')
plt.title('Model RMSE Comparison by Time Period')
plt.ylabel('RMSE')
plt.show()

# R2
plt.figure(figsize=(12,6))
ax = sns.barplot(data=results_df, x='Period', y='R2', hue='Model')
plt.title('Model R² Comparison by Time Period')
plt.ylabel('R²')
plt.legend(loc='lower right')
plt.show()





import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator,DayLocator
import pandas as pd

models = ['LSTM', 'BiLSTM', 'CNN-LSTM', 'CNN-BiLSTM']
periods = ['data_1yr', 'data_5yr', 'data_10yr', 'data_20yr']

for period_name in periods:
    plt.figure(figsize=(16, 12))
    plt.suptitle(f'Period: {period_name}', fontsize=18)


    period_results = [res for res in results if res['Period'] == period_name]

    for i, model_name in enumerate(models):
        ax = plt.subplot(2, 2, i + 1)


        res = next((r for r in period_results if r['Model'] == model_name), None)
        if res is None:
            continue

        res['date_test'] = pd.to_datetime(res['date_test'])  # 確保是 datetime

        ax.plot(res['date_test'], res['y_test'], label='Actual', marker='o', markersize=1)
        pred_label = f"Predicted (RMSE={res['RMSE']:.2f}, R²={res['R2']:.2f})"
        ax.plot(res['date_test'], res['y_pred'], label=pred_label, marker='x', markersize=1)

        duration_years = (res['date_test'][-1] - res['date_test'][0]).days / 365.25

        if duration_years > 15:
          ax.xaxis.set_major_locator(YearLocator(5)) 
        elif duration_years > 8:
          ax.xaxis.set_major_locator(YearLocator(1))  
        elif duration_years > 1:
          ax.xaxis.set_major_locator(MonthLocator(interval=6))  
        else:
          ax.xaxis.set_major_locator(MonthLocator(interval=1))  

        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', labelrotation=45)

        ax.set_title(f"{model_name}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        plt.xticks(rotation=45)
        ax.legend(loc='lower right', fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

