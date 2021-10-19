import pandas as pd
import MetaTrader5 as mt5
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from datetime import datetime
from pytz import timezone
import time
from Ordens2 import trader 
import pyautogui


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
  new_df1 = []
  for d in range(len(data)-1):
     new_df.append(sigmoid(data['close'][d]-data['open'][d]))
  new_df1.append([new_df])
  novo_df = pd.DataFrame(new_df1,columns=['Feature'])
  return novo_df

ativos = ['EURUSD','PETR4','VALE3','ITUB3','IBOV']

windons=3000

new_model = tf.keras.models.load_model('M5_model.h5')
sticker = get_ohlc(ativos[0],mt5.TIMEFRAME_H1,windons+1)
dir = 2
pos = '-'
new_vlr = 0
lucro = 0
lista = [0,5,10,15,20,25,30,35,40,45,50,55]
while True:
    data_e_hora_atuais = datetime.now()
    fuso_horario = timezone('America/Sao_Paulo')
    hora = data_e_hora_atuais.astimezone(fuso_horario)
    #hora = hora.strftime('%H')
    minutos= hora.strftime('%M')
    sticker = get_ohlc(ativos[0],mt5.TIMEFRAME_H1,windons+1)
    vlr = sticker['close'][-1:].values[0]
    df = process(sticker,windons)
    X = np.hstack((df['Feature'])).reshape(-1,windons)

    if int(minutos) in lista:
        pred = np.argmax(new_model.predict(X))
        #print(pred)
        if pred == 0 and dir != 0:
            dir = pred
            pyautogui.click(x=1902, y=902)
            time.sleep(1)
            pyautogui.click(438,175)
            print("Ordem de venda emitida no valor de R$ "+str(vlr),end='\r')
            pos = 'Venda'
            lucro = vlr - new_vlr
            new_vlr = vlr
        if pred == 1 and dir != 1:
            dir = pred
            pyautogui.click(x=1902, y=902)
            time.sleep(1)
            pyautogui.click(593, 175)
            print("Ordem de compra emitida no valor de R$ "+str(vlr),end='\r')
            pos = 'Compra'
            lucro = vlr - new_vlr
            new_vlr = vlr
        if pred == 2 and dir != 2:
            dir = pred
            print("Sem ação",end='\r')
        else:
            print('Operação fechou com Lucro de: R$ '+str(lucro)+' // Nova ordem de '+pos+' emitida no valor de R$ '+str(vlr),end='\r')
    else:
        print('Ordem atual: '+str(dir)+' // Bot ir operar a cada 5 min',end='\r')
    time.sleep(1)