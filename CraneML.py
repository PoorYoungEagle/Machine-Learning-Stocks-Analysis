from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from sklearn.linear_model import LinearRegression
import pandas_ta as ta

'''

Main code for Machine Learning
Uses indicators and Linear Regression to learn the data

'''

#Inputs given ticker and returns data associated to that stock
def ticker(ticker="NVDA"):
  df = si.get_data(ticker)
  open_price = df['open']
  close_price = df['close']
  volumes = df['volume']
  high_price = df['high']
  low_price = df['low']
  DATA_LEN = 300

  open_prices = df['open']
  close_prices = df['close']
  high_prices = df['high']
  low_prices = df['low']

  close_prices = close_prices[len(close_prices) - DATA_LEN:len(close_prices)].to_list()
  open_prices = open_prices[len(open_prices) - DATA_LEN:len(open_prices)].to_list()
  volumes = volumes[len(volumes) - DATA_LEN:len(volumes)].to_list()
  high_prices = high_prices[len(high_prices) - DATA_LEN:len(high_prices)].to_list()
  low_prices = low_prices[len(low_prices) - DATA_LEN:len(low_prices)].to_list()

  return df, open_prices, close_price, close_prices, high_prices, low_prices, volumes

DATA_LEN = 300
df, open_prices, close_price, close_prices, high_prices, low_prices, volumes = ticker()

#Just plots the open prices
def plot_open_prices(open_prices):
  plt.style.use('dark_background')
  plt.figure(figsize=(10, 6))
  plt.title("Open prices")
  plt.xlabel("Days after")
  plt.ylabel("Price")
  plt.plot(open_prices)
  plt.grid(True)
  plt.show()
  return plt.gcf()
plot_open_prices(open_prices)

#Just plots the close prices
def plot_close_prices(close_prices):
  plt.figure(figsize=(18, 5))
  plt.title("Close prices", fontsize=15)
  plt.xlabel("Days after", fontsize=12)
  plt.ylabel("Price", fontsize=12)
  plt.plot(close_prices, 'cyan', label='Close Price')
  plt.axhline(y=close_prices[len(close_prices) - 1], label=f'Latest Close Price: {close_prices[len(close_prices) - 1]}')
  plt.legend()
  #plt.savefig("graphed-results/close-price")
  plt.grid(True)
  plt.show()
  return plt.gcf()
plot_close_prices(close_prices)


model = LinearRegression()

#Placeholder function which is to be exported from Cranelit.py
def Progress_Update(Placeholder):
    pass

#Main Machine Learning code function using Linear Regression model
def MLModel(Progress_Update, epochs=100):

  #Uses RSI, EMA and SMA indicators. Other indicators can be brought in by adding another pandas_ta line with the indicator and then placing the name into the 'X' pandas dataframe
  df['RSI'] = ta.rsi(close_price, length=14)
  df['EMA'] = ta.ema(close_price, length=14)
  df['SMA'] = ta.sma(close_price, length=14)

  df['Close_Lag1'] = df['close'].shift(1)

  df['Target'] = (close_price.shift(-1) > close_price).astype(int)

  X=df[['RSI','EMA','SMA']].dropna()
  print(type(X))
  Y=df['Target'].loc[X.index]
  scaler = MinMaxScaler()
  X_scaled = scaler.fit_transform(X)

  #Machine Learning Code
  X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1)
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(256, input_shape=(X_train.shape[1],), activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  initial_learning_rate = 0.001
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
  )
  optimizer = Adam(learning_rate=lr_schedule,clipvalue=1.0)
  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

  #Progress slider for streamlit
  Accuracy = []
  Val_Accuracy = []

  for epoch in range(epochs):
    history=model.fit(X_train, Y_train, epochs=1, validation_data=(X_test, Y_test))

    Accuracy.extend(history.history['accuracy'])
    Val_Accuracy.extend(history.history['val_accuracy'])

    progress = (epoch + 1) / epochs
    Progress_Update(progress)

  #Plotting the accuracy on the graph with relation to Epoch and Accuracy
  loss,accuracy = model.evaluate(X_test, Y_test)
  print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}")
  plt.plot(Accuracy, label='Accuracy')
  plt.plot(Val_Accuracy, label='Validation Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0, 1])
  plt.xlim([1,epochs])
  plt.legend(loc='lower right')
  plt.show()
  return plt.gcf(), loss, accuracy, history

Placehold,loss,accuracy,history = MLModel(Progress_Update,1)

"""def Accuracy_Graph():
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}")
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    return plt.gcf()"""