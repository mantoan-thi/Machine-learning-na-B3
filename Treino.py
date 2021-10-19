import pandas as pd
import MetaTrader5 as mt5
import math
import numpy as np
import tensorflow  as tf
from sklearn.model_selection import train_test_split

# conecte-se ao MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()


def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + math.exp(gamma))
    return 1 / (1 + math.exp(-gamma))

def get_ohlc(ativo,timeframe, n):
    ativo = mt5.copy_rates_from_pos(ativo,timeframe,0,n)
    ativo = pd.DataFrame(ativo)
    ativo['time'] = pd.to_datetime(ativo['time'],unit='s')
    ativo.set_index('time',inplace=True)
    ativo.reset_index(inplace=True)
    return ativo

def process(data,windons):
  new_df = []
  for d in range(len(data)):
    dif = []
    dif_sig = []
    for i in range(d,d+windons):
        dif.append(data['close'][i]-data['open'][i])
        dif_sig.append(sigmoid(data['close'][i]-data['open'][i]))
    if np.sum(dif) > 0:
        new_df.append([dif_sig,1])
    elif np.sum(dif) < 0:
        new_df.append([dif_sig,0])
    else:
        new_df.append([dif_sig,2])
    if d == (data.shape[0]-windons):
        break
  novo_df = pd.DataFrame(new_df,columns=['Feature','Tag'])
  novo_df['Tag'] = novo_df['Tag'].shift(-1)
  novo_df.dropna(inplace=True)
  novo_df['Tag'] = novo_df['Tag'].map(int)
  return novo_df


windons = 40

ativos = ['EURUSD','PETR4','VALE3','ITUB3','IBOV']

sticker = get_ohlc(ativos[0],mt5.TIMEFRAME_D1,5000)

df = process(sticker,windons)
X = np.hstack((df['Feature'])).reshape(-1,windons)
y = np.hstack(df['Tag'])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=64,activation='relu',input_shape=(windons,)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=128,activation = "relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=128,activation = "relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=64,activation = "relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=3,activation='softmax'))
model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'],
            run_eagerly=True)
            

model.summary()

model.fit(X_train,y_train,epochs=100,verbose=1)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))

# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('M5_model.h5')
