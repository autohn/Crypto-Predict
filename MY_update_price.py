# %%

#бинанс
from cmath import nan
from binance.client import Client
from binance.exceptions import BinanceAPIException
from helpers.handle_creds import (
    load_correct_creds, test_api_key
)

from helpers.parameters import (
    parse_args, load_config
)

#часть1
import numpy as np 
import pandas as pd 
import time
from datetime import datetime, timedelta, date

import importlib
import perdict_VSCode
importlib.reload(perdict_VSCode)
from perdict_VSCode import *

#часть2
import os
import threading
import keyboard
import logging
import json
import pprint
import math



# %%



crypto_name = 'Bitcoin'

#file_name = 'btcusd_my.csv'realtime_test.csv
file_name = 'realtime_test.csv'

is_realtime_set = 'Y'

#df_o = load_dataset(file_name, '1T')
df_last = load_dataset(file_name, '1T', is_realtime_set)

print_dataset(df_last, crypto_name)

# %%

col_set, colmns = get_col_set(df_last)

#df, train, test, train_sep = data_partitioning(df_o, crypto_name, col_set)

bors =  'N'

realtime_set, windowX, windowY, v_min, v_max = create_dataset(None, None, None, col_set, colmns, bors, is_realtime_set, df_last)

print(len(realtime_set))



# %%
model, batch_size, predict_batch_size, model_type = create_model(realtime_set, col_set, windowY) #TODO переделать, вызывается просто для получения констант


model = load_model('saved_models/перрвый_вырос_упал')

# %%

prediction, prediction0 = predict(model, v_min, v_max, realtime_set, predict_batch_size, col_set, model_type)
#обучилась, сохранилась теперь ее запустить с новыми параметрами реалтайм_сет, а потом его в цикле

print(prediction)
print(prediction0)

# %%



def realtime_predict():
    df_last = load_dataset(file_name, '1T', is_realtime_set)
    realtime_set, windowX, windowY, v_min, v_max = create_dataset(None, None, None, col_set, colmns, bors, is_realtime_set, df_last)
    prediction, prediction0 = predict(model, v_min, v_max, realtime_set, predict_batch_size, col_set, model_type)
    print("Предсказанная цена: ", prediction[0].astype(float))
    return prediction[0].astype(float)



# %%
df_alt = df_o.fillna(method='ffill')
df_o = df_alt.fillna(method='bfill')

#print(df_o.head())
print("Head без nan: ", df_o.head())
print_dataset(df_o, crypto_name)
#df_o.to_csv(file_name)



# %%
'''
def dateparse (time_in_secs):    
    return datetime.fromtimestamp(float(time_in_secs)/1000)

df_file = pd.read_csv(file_name, sep=',', parse_dates=[0], date_parser=dateparse, index_col='time')

print(df_file.head())

'''
#df_file = pd.read_csv(file_name, sep=',', parse_dates=[0], index_col='time')
# %%
# df_file.to_csv('btcusd_my.csv')


# %%

TEST_MODE = True

DEFAULT_CREDS_FILE = 'creds.yml'

parsed_creds = load_config(DEFAULT_CREDS_FILE) #TODO проверить че там как работает

access_key, secret_key = load_correct_creds(parsed_creds)

client = Client(access_key, secret_key)

api_ready, msg = test_api_key(client, BinanceAPIException)
if api_ready is not True:
    print('api_key error')

# %%
print(client.get_asset_balance(asset='SHIB'))

print(client.get_symbol_ticker(symbol="BTCUSDT")['price'])


# %%
def enrich(orig_dframe, prew_dframe = None):
    #TODO добавить гугл трендинг
    mdframe = pd.DataFrame(columns = [ "open", "close", "high", "low", "volume", "day_of_week_num", "day_of_month_num", "change", "google_tranding"] )
    mdframe = mdframe.append(orig_dframe)

    for el in ["close", "high", "low", "volume", "google_tranding"]:
        if math.isnan(mdframe.loc[mdframe.index[0], el]):
            mdframe.loc[mdframe.index[0], el] = 0
    if math.isnan(mdframe.loc[mdframe.index[0], "day_of_week_num"]):
        mdframe.loc[mdframe.index[0], "day_of_week_num"] = mdframe.index.dayofweek
    if math.isnan(mdframe.loc[mdframe.index[0], "day_of_month_num"]):
        mdframe.loc[mdframe.index[0], "day_of_month_num"] = mdframe.index.day
    if math.isnan(mdframe.loc[mdframe.index[0], "change"]):
        if prew_dframe is not None:
            change = mdframe.loc[mdframe.index[0], "open"] - prew_dframe.loc[prew_dframe.index[0], "open"]
            mdframe.loc[mdframe.index[0], "change"] = 1 if change > 1 else 0
        else:
            mdframe.loc[mdframe.index[0], "change"] = 0
    #mdframe = mdframe.fillna(0)

    return mdframe


print(enrich(pd.DataFrame({'open': 123, 'close': np.NaN, 'high': np.float(1), 'low': 456, 'volume': 0}, index={datetime.fromtimestamp(int(time.time()))}), pd.DataFrame({'open': 120, 'close': np.NaN, 'high': 345, 'low': 456, 'volume': 0}, index={datetime.fromtimestamp(int(time.time()))})))



# %%
will_work = 1

total = 0
update_total_lock = threading.Lock()



def update_total():
    global df_file
 
    with update_total_lock:
        btc_price = float(client.get_symbol_ticker(symbol="BTCUSDT")['price'])
        df_new_row = pd.DataFrame({'open': btc_price, 'close': btc_price, 'high': btc_price, 'low': btc_price, 'volume': 0}, index={datetime.fromtimestamp(int(time.time()))})

        df_e_row = enrich(df_new_row) #TODO передавать прошлый элемент чтобы работал change 

        #df_file = df_new_row
        #df_file = df_file.append(df_new_row)
        #df_file.index.name = 'time' #TODO а есть ли способ это делать сразу, а не отдельной строкой

        df_e_row.to_csv('realtime_test.csv', mode='a', header=False)




        last_price = df_e_row['open'][0].astype(float)

        print("Последняя цена: ", last_price)

        pred_price = realtime_predict()

        if pred_price > last_price:
            buy('BTCUSDT', 0.05)
        else:
            sell('BTCUSDT', 0.05)

        # df_file.to_csv('realtime_test.csv')#file_name)
        
    print ('---update_total')

def start_loop():
    global will_work
    will_work = 1

def stop_loop():
    global will_work
    will_work = 0

keyboard.add_hotkey('Ctrl + 1', start_loop)
keyboard.add_hotkey('Ctrl + 2', stop_loop)


last_times = {'update_prise_in_csv': time.time() - 60}


true = True #

while true:
    if(will_work):
        if(time.time() - last_times['update_prise_in_csv'] > 10):
            last_times['update_prise_in_csv'] = time.time()
            print(time.time())
            my_thread = threading.Thread(target=update_total, args=())
            my_thread.start()

    else:
        print('0')
   

keyboard.wait()


# %%


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename='my_log.log', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
logging.error('And non-ASCII stuff, too, like Øresund and Malmö')



# %%

symbol = 'ZECUSDT'
quantity = 0.05

def buy(symbol, quantity):
    if TEST_MODE:
       logging.info('Пытаемся купить '+str(symbol)+' на '+str(quantity)) #TODO опять не работает
       print('Пытаемся купить '+str(symbol)+' на '+str(quantity))
    else:
        try:
            buy_limit = client.create_order(
                symbol = symbol,
                side = 'BUY',
                type = 'MARKET',
                quantity = quantity
            )
        except Exception as e:
            print(e)
            logging.error(e)
        else:
            order = client.get_all_orders(symbol=symbol, limit=1)
            
            # binance sometimes returns an empty list, the code will wait here until binance returns the order
            while order == []:
                print('Ждем тормоза бинанса')
                order = client.get_all_orders(symbol=symbol, limit=1)
                time.sleep(1)

            else:
                print('Ордер на покупку получен')
            logging.info('Выставили ордер на покупку: ', order)

#INFO:root:[{'symbol': 'ZECUSDT', 'orderId': 1327014573, 'orderListId': -1, 'clientOrderId': '1tFFXsSeQjo7H76z63CHcb', 'price': '0.00000000', 'origQty': '0.05000000', 'executedQty': '0.05000000', 'cummulativeQuoteQty': '11.37500000', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'MARKET', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': 1638110411357, 'updateTime': 1638110411357, 'isWorking': True, 'origQuoteOrderQty': '0.00000000'}]

buy(symbol, quantity)


# %%


def sell(symbol, quantity):
    if TEST_MODE:
       logging.info('Пытаемся продать '+str(symbol)+' на '+str(quantity))
       print('Пытаемся продать '+str(symbol)+' на '+str(quantity))
    else:
        try:
            sell_limit = client.create_order(
                symbol = symbol,
                side = 'SELL',
                type = 'MARKET',
                quantity = quantity
            )
        except Exception as e:
            print(e)
            logging.error(e)
        else:
            order = client.get_all_orders(symbol=symbol, limit=1)
            
            while order == []:
                print('Ждем тормоза бинанса')
                order = client.get_all_orders(symbol=symbol, limit=1)
                time.sleep(1)

            else:
                print('Ордер на продажу получен')
            logging.info('Выставили ордер на продажу: ', order)

#INFO:root:[{'symbol': 'ZECUSDT', 'orderId': 1327014573, 'orderListId': -1, 'clientOrderId': '1tFFXsSeQjo7H76z63CHcb', 'price': '0.00000000', 'origQty': '0.05000000', 'executedQty': '0.05000000', 'cummulativeQuoteQty': '11.37500000', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'MARKET', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': 1638110411357, 'updateTime': 1638110411357, 'isWorking': True, 'origQuoteOrderQty': '0.00000000'}]

sell(symbol, quantity)
# %%

symbol_price = client.get_symbol_ticker(symbol=symbol)
print(symbol_price)

# %%
prices = client.get_all_tickers()

print(prices)

# %%

info = client.get_symbol_info('ZECUSDT')
step_size = info['filters'][2]['stepSize']
lot_size = step_size.index('1') - 1


print(lot_size)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(info)

# %%

a = 0.321123
a = float('{:.{lot_size}f}'.format(a, lot_size=lot_size))


print(a)



# %%


def sf(num):
    return float('{:.2f}'.format(num))



def moving_averages(df_file):
    short_mean = sf(df_file.iloc[-10:]['open'].mean())
    long_mean = sf(df_file.iloc[-30:]['open'].mean())

    mean_status = 1 if short_mean > long_mean else 0

    past_short_mean = sf(df_file.iloc[-11:-1]['open'].mean())
    past_long_mean = sf(df_file.iloc[-31:-1]['open'].mean())

    past_mean_status = 1 if past_short_mean > past_long_mean else 0


    if (mean_status > past_mean_status):
        buy(symbol, quantity)
    elif (mean_status < past_mean_status):
        sell(symbol, quantity)
    else:
        logging.info('Ждем '+str(mean_status)+' sm: '+str(short_mean)+' lm: '+str(long_mean)+' psm: '+str(past_short_mean)+' plm: '+str(past_long_mean))

moving_averages(df_last)

# %%
