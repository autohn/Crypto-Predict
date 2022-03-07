# %%

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD

#from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
import tensorflow.keras.models
import time
from madgrad import MadGrad
from enum import Enum
from pytrends.request import TrendReq
from pytrends import dailydata




# %% ###################################################################################################################################
def load_dataset(file_name, resample_rate, realtime_set = None):
    #Convert date
    '''
    def to_datetime(df_o):
        date = datetime.strptime(df_o, '%d.%m.%Y')
        return date.strftime("%Y-%m-%d")
    '''
    
    def dateparse (time_in_secs):    
        return datetime.fromtimestamp(float(time_in_secs)/1000)
    # -2300240
    if realtime_set:
        df_o = pd.read_csv(file_name, sep=',', parse_dates=[0], index_col='time').iloc[-4320:]
    else:
        df_o = pd.read_csv(file_name, sep=',', parse_dates=[0], index_col='time').iloc[-23002:]
    #df_o = pd.read_csv('PriceTesla.csv', sep=';', parse_dates=True, index_col='Date').iloc[:]


    #df_o['Date'] = df_o['Date'].apply(lambda x: to_datetime(x))

    if realtime_set:
        print("Realtime cет говтов, размерность: ", df_o.shape)
    else:
        print("Сет загружен, оригинальная размерность: ", df_o.shape)

        df_o = df_o.resample(resample_rate, label='right').agg({'open': np.mean, 'close': np.mean, 'high': np.max, 'low': np.min, 'volume': np.sum, 'day_of_week_num': np.max, 'day_of_month_num': np.max, 'change': np.max, 'google_tranding': np.mean})

        print("Ресемпл: ", resample_rate, " размерность после ресэмпла: ", df_o.shape)

    return df_o

#df_o.dropna(how='all', axis=1, inplace=True)


def mass_load_dataset(df_o):
    #df_o['day_of_week'] = df_o['open'].index.day_name()
    df_o['day_of_week_num'] = df_o['open'].index.dayofweek
    df_o['day_of_month_num'] = df_o['open'].index.day

    
    change = df_o.diff()['open']
    change[change > 0] = 1
    change[change <= 0] = 0
    change[np.isnan(change)] = 0

    df_o['change'] = change
    df_o = df_o.iloc[1:]
    
    print("Размерность: ", df_o.shape)
    return df_o

# %% ###################################################################################################################################

def fill_nan(df_o):
    #arraydf[np.isnan(arraydf)] = arraydf[np.isnan(arraydf)] np.nan

    #todo сделать как-то менее убого и сделать нормальную обработку других столбиков

    time_range = df_o['open'].index[-1] - df_o['open'].index[-2]

    print("Временно промежуток: ", time_range)

    before_nan = 0
    count_nan = 0
    first_nan = 0
    global_count_nan = 0

    for index, row in df_o.iterrows():
        if(np.isnan(row['open'])):
            if(count_nan == 0):
                first_nan = index
                print(index)
                before_nan = df_o['open'][index-time_range]
            count_nan += 1
            global_count_nan += 1
        else:
            if(count_nan > 0):         
                step = (row['open'] - before_nan)/count_nan
                for i in range(0, count_nan):
                    df_o['open'][first_nan + i * time_range] = before_nan + step * i
                    df_o['high'][first_nan + i * time_range] = before_nan + step * i
                    df_o['low'][first_nan + i * time_range] = before_nan + step * i
                first_nan = 0
                count_nan = 0

    print("Сколько было удалено nan: ", global_count_nan)          
    #print(df_o)

    return df_o



# %% ###################################################################################################################################
def get_trends(crypto_name, s_y, s_m, e_y, e_m, geo):
    pytrend = TrendReq()

    #2014, 10
    google_tranding_all = dailydata.get_daily_data(crypto_name, s_y, s_m, e_y, e_m, geo = geo)

    #не используем по часам потому что походу там масштаб только по неделям
    #crypto_name_h = [crypto_name]
    #google_tranding_all = pytrend.get_historical_interest(crypto_name_h, year_start=2016, month_start=4, day_start=1, hour_start=0, year_end=2016, month_end=11, day_end=1, hour_end=0, cat=0, geo='', gprop='', sleep=1)

    google_tranding_all.to_csv('google_tranding_all_d_' + str(s_y) + '_' + str(s_m) + '_' + str(e_y) + '_' + str(e_m) + '.csv')

    return 0

#get_trends(crypto_name, 2014, 10, 2021, 6, '')


# %% ###################################################################################################################################
def load_trends(crypto_name, file_name, df_o):
    #google_tranding = google_tranding.rename(columns={"cinema_monthly": "test"})

    google_tranding_all = pd.read_csv(file_name, sep=',', parse_dates=True, index_col='date')

    time_interval = google_tranding_all.index[-1] - google_tranding_all.index[-2]

    if(time_interval.days == 1):
        google_tranding_all = google_tranding_all.resample('1H', label='right').pad()

    google_tranding_all = google_tranding_all.rename_axis("time")
    google_tranding = (google_tranding_all.iloc[:, 4]).rename('google_tranding')

    if (crypto_name in df_o.columns): df_o = df_o.drop(crypto_name, 1) #чтобы не сломалось если несколько раз вызвать ячейку
    df_o = pd.merge(df_o, google_tranding, how='left', on='time')

    print("Размерность с трендами: ", df_o.shape)

    print("Head с трендами: ", df_o.head())
    return df_o




# %% ###################################################################################################################################

def data_to_sin(df_o):
    x = np.arange(df_o.shape[0])*(20*np.pi/1000)
    dataset = (np.sin(x) * 2 * x * np.cos(2 * x))[:,None]
    df_o['open'] = dataset
    #[:,None]


    change = df_o.diff()['open']
    change[change > 0] = 1
    change[change <= 0] = 0
    change[np.isnan(change)] = 0
    #print(change)
    df_o['change'] = change
    df_o = df_o.iloc[1:]
    
    
    print(dataset)
    print(df_o)
    print(type(df_o))
    return df_o

#df_o = data_to_sin(df_o)


# %% ###################################################################################################################################
def print_dataset(df_o, crypto_name):
    df_o['open'] = df_o['open'].astype(float)

    plt.figure(figsize=(20,7))
    plt.plot(df_o['open'].index, df_o['open'].values, label = 'BTC Price', color = 'red', marker='.', linestyle='')
    if(crypto_name != ''): plt.plot(df_o['google_tranding'].index, df_o['google_tranding'].values*1000, label = 'Google trands', color = 'green', marker='.', linestyle='')
    #plt.xticks(np.arange(100,df_o.shape[0],200))
    plt.xlabel('time')
    plt.ylabel('open ($)')
    plt.legend()
    plt.show()




    plt.figure(figsize=(20,7))
    plt.plot(df_o['open'].index, df_o['open'].values, label = 'BTC Price', color = 'red')
    if(crypto_name != ''): plt.plot(df_o['google_tranding'].index, df_o['google_tranding'].values*1000, label = 'Google trands', color = 'green')
    #plt.xticks(np.arange(100,df_o.shape[0],200))
    plt.xlabel('time')
    plt.ylabel('open ($)')
    plt.legend()
    plt.show()

    return 0



# %% ###################################################################################################################################

def old_data_partitioning(df_o):
    col_set = [df_o.columns.tolist().index('open'), df_o.columns.tolist().index('volume'), df_o.columns.tolist().index('day_of_week_num')]
    #col_set = [df_o.columns.tolist().index('open'), df_o.columns.tolist().index('volume'), df_o.columns.tolist().index('day_of_week_num'), df_o.columns.tolist().index(crypto_name)]

    test_size = 200
    real_test_size = 10
    train_sep = df_o.shape[0] - test_size - real_test_size
    test_sep = train_sep + test_size

    train = df_o.iloc[:train_sep, col_set].values
    test = df_o.iloc[train_sep:test_sep, col_set].values
    real_test = df_o.iloc[test_sep:, col_set].values


    df = df_o[:test_sep]

    df.shape


    print(type(train))
    print(train.shape)
    #print(df)
    #print(df_o[:-100])
    return df, train, test, real_test, test_size, real_test_size, train_sep, test_sep, col_set 

#df, train, test, real_test, test_size, real_test_size, train_sep, test_sep, col_set = old_data_partitioning(df_o)

# %% ###################################################################################################################################

def old_train_create(train):
    sc = MinMaxScaler(feature_range = (0, 1))
    train_scaled = sc.fit_transform(train)


    #np.random.seed(42)
    global windowX
    global windowY
    windowX = 100
    windowY = 10



    def many_to_one_off_one(train_scaled):
        #много минут в 1 со смещением в 1
        global windowY
        windowY = 1
        X_train = []
        y_train = []

        for i in range(windowX, train_sep):
            X_train_ = np.reshape(train_scaled[i-windowX:i], (windowX, len(col_set)))
            X_train.append(X_train_)
            y_train.append(train_scaled[i, 0])

        X_train = np.stack(X_train)
        y_train = np.stack(y_train)
        return X_train, y_train


    def many_to_one_off_window(train_scaled):
        #много минут в 1 со смещением в окно
        global windowY
        windowY = 1
        X_train = []
        y_train = []

        i = windowX

        while i < train_sep:
            X_train_ = np.reshape(train_scaled[i-windowX:i, 0], (windowX, 1))
            X_train.append(X_train_)
            y_train.append(train_scaled[i, 0])
            i += windowX
            #if(i>30): break


        X_train = np.stack(X_train)
        y_train = np.stack(y_train)
        return X_train, y_train


    def seq2seq_off_one(train_scaled):
        #много минут в много минут со смещением в 1
        X_train = []
        y_train = []

        for i in range(windowX+windowY, train_sep+1):
            X_train_ = np.reshape(train_scaled[i-windowX-windowY:i-windowY], (windowX, len(col_set)))
            X_train.append(X_train_)
            y_train.append(train_scaled[i-windowY:i, 0])

        X_train = np.stack(X_train)
        y_train = np.stack(y_train)

        return X_train, y_train



    X_train, y_train = many_to_one_off_one(train_scaled)


    print(X_train.shape)
    print(y_train.shape)


    #print(max(map(max, y_train))) 
    #print(X_train[1290])
    return X_train, y_train, sc

#X_train, y_train, sc = old_train_create(train)

# %% ###################################################################################################################################


def old_test_create(train, test):
    df_volume = np.vstack((train, test))
    #print(test)

    inputs = df_volume[df_volume.shape[0] - test.shape[0] - windowX:] #беремс конца элементов в количестве тестовых + окно
    inputs = inputs.reshape(-1,len(col_set)) #фактически просто проверка размерности
    inputs = sc.transform(inputs) #типа нормальизованные значения?

    num_2 = df_volume.shape[0] - train_sep + windowX + 1


    X_test = []
    y_test = []

    for i in range(windowX+windowY, num_2): #добавляет по 60 прошлых на каждый пример, то есть предсказываем только один шаг
        X_test_ = np.reshape(inputs[i-windowX-windowY:i-windowY], (windowX, len(col_set)))
        X_test.append(X_test_)
        y_test.append(inputs[i-windowY:i, 0])


    X_test = np.stack(X_test)
    y_test = np.stack(y_test)
    print(X_test.shape) #0.44925722 0.45019758
    print(y_test.shape) 
    #X_test = X_test[:25]  #[:25] временный костыль для stateful ломает вывод
    #test = test[:25]

    return X_test, y_test


#X_test, y_test = old_test_create(train, test)



# %% ###################################################################################################################################



def get_col_set(df_o):
    class colmns(Enum):
        open = df_o.columns.tolist().index('open')
        close = df_o.columns.tolist().index('close')
        volume = df_o.columns.tolist().index('volume')
        high = df_o.columns.tolist().index('high')
        low = df_o.columns.tolist().index('low')
        day_of_week_num = df_o.columns.tolist().index('day_of_week_num')
        day_of_month_num = df_o.columns.tolist().index('day_of_month_num')
        change = df_o.columns.tolist().index('change')
        google_tranding = df_o.columns.tolist().index('google_tranding')

    #TODO мб использовать df.iloc[0:10, df.columns.get_loc('column_name')] и вообще нормально переделать

    col_set = [colmns['open'].value, colmns['close'].value, colmns['volume'].value, colmns['high'].value, colmns['low'].value, colmns['day_of_week_num'].value, colmns['day_of_month_num'].value, colmns['change'].value, colmns['google_tranding'].value]

    #col_set = [colmns['open'].value, colmns['volume'].value, colmns['high'].value, colmns['low'].value, colmns['day_of_week_num'].value, colmns['day_of_month_num'].value, colmns['change'].value, df_o.columns.tolist().index(crypto_name)]

    #col_set = [colmns['open'].value, colmns['volume'].value, colmns['high'].value, colmns['low'].value, colmns['day_of_week_num'].value, colmns['day_of_month_num'].value]

    #col_set = [colmns['open'].value, colmns['volume'].value, colmns['high'].value, colmns['low'].value, colmns['day_of_week_num'].value, colmns['day_of_month_num'].value, df_o.columns.tolist().index(crypto_name)]
    
    #col_set = [df_o.columns.tolist().index('open'), df_o.columns.tolist().index('volume'), df_o.columns.tolist().index('day_of_week_num'), df_o.columns.tolist().index(crypto_name)]

    return col_set, colmns




def data_partitioning(df_o, crypto_name, col_set):

    test_size = 500
    real_test_size = 10
    train_sep = df_o.shape[0] - test_size - real_test_size
    test_sep = train_sep + test_size

    train = df_o.iloc[:train_sep, col_set].values
    test = df_o.iloc[train_sep:test_sep, col_set].values
    real_test = df_o.iloc[test_sep:, col_set].values


    df = df_o[:test_sep]

    df.shape

    print(type(train))
    print("Размер тренировочного сета:", train.shape)


    return df, train, test, train_sep



# %% ###################################################################################################################################
def create_dataset(train, test, train_sep, col_set, colmns, bors, is_realtime_set = None, df_last = None):
    def create_set_dataframe(data_set, separator, step, windowX, windowY, colmns, bors, is_realtime_set = None):
        #TODO train to input
        X_input = np.empty((0,62,len(col_set)), float)
        y_input = np.empty((0,windowY), float)

        #for index, row in data_set.iterrows():
            #print(index, '  ', row)
            #break
        #print(data_set[0:1])

        def stack_datagrame_range(x):
            return np.asarray([x[:,colmns['open'].value].mean(axis=0), x[:,colmns['close'].value].sum(axis=0), x[:,colmns['volume'].value].sum(axis=0), x[:,colmns['high'].value].max(axis=0), x[:,colmns['low'].value].min(axis=0), x[:,colmns['day_of_week_num'].value].min(axis=0), x[:,colmns['day_of_month_num'].value].min(axis=0), x[:,colmns['change'].value].mean(axis=0), x[:,colmns['google_tranding'].value].mean(axis=0)])
            #return np.asarray([x[:,colmns['open'].value].mean(axis=0), x[:,colmns['volume'].value].sum(axis=0), x[:,colmns['high'].value].max(axis=0), x[:,colmns['low'].value].min(axis=0), x[:,colmns['day_of_week_num'].value].min(axis=0), x[:,colmns['day_of_month_num'].value].min(axis=0), x[:,colmns['change'].value].mean(axis=0), x[:,df.columns.tolist().index(crypto_name)].mean(axis=0)])
        
        for i in range(windowX+windowY, separator+1, step):
            #X_train_ = np.reshape(train_scaled[i-windowX:i], (windowX, len(col_set)))
            biasm = 0
            biasd = 0
            biash = 0
            X_input_ = np.empty((0,len(col_set)), float)
            
            #print(data_set[0:1])
            
            for m in range(10): #0-5
                skipm = 24 * 15
                biasm = m * skipm
                x = data_set[i-windowX-windowY+biasm:i-windowX-windowY+biasm+skipm]
                x = stack_datagrame_range(x)
                #x = pd.DataFrame(x).agg({0: np.mean, 1: np.sum, 2: np.max, 3: np.min, 4: np.min})
                X_input_ = np.append(X_input_, np.expand_dims(x, axis=0), axis=0)
                #print(i-windowX-windowY+biasm, i-windowX-windowY+biasm+skipm)

            for d in range(29):
                skipd = 24
                biasd = biasm + skipm + d * skipd
                x = data_set[i-windowX-windowY+biasd:i-windowX-windowY+biasd+skipd]
                x = stack_datagrame_range(x)
                X_input_ = np.append(X_input_, np.expand_dims(x, axis=0), axis=0)
                #print(i-windowX-windowY+biasd,i-windowX-windowY+biasd+skipd)
                #print(np.argwhere(np.isnan(x)))
            for h in range(23):
                skiph = 1
                biash = biasd + skipd + h
                x = data_set[i-windowX-windowY+biash:i-windowX-windowY+biash+skiph]
                x = stack_datagrame_range(x)
                X_input_ = np.append(X_input_, np.expand_dims(x, axis=0), axis=0)
                #print(i-windowX-windowY+biash,i-windowX-windowY+biash+skiph)
                #print(np.argwhere(np.isnan(x)))

            X_input = np.append(X_input, np.expand_dims(X_input_, axis=0), axis=0)
            #print(np.argwhere(np.isnan(X_input)))
            if not is_realtime_set:
                if(bors != 'Y'): #чо за акроним забыл, bigger or smaller?
                    y_input = np.append(y_input, np.expand_dims(data_set[i-windowY:i, colmns['open'].value], axis=0), axis=0)
                else:
                    y_input = np.append(y_input, np.expand_dims(data_set[i-windowY:i, colmns['change'].value], axis=0), axis=0)
            #break


        #print(x_last+windowX)
        if is_realtime_set:
            return X_input
        else:
            X_input = np.stack(X_input)
            y_input = np.stack(y_input)
            return X_input, y_input




    windowY = 10
    windowX = 24 * 30 * 6 #4320

    #train
    #sc = MinMaxScaler(feature_range = (0, 1))
    #train_scaled = sc.fit_transform(train)
    
    #print(type(train_scaled))
    #train_scaled = pd.DataFrame(sc.fit_transform(train),columns = train.columns)

    #train_scaled = train #.copy()
    #train_scaled[:] = sc.fit_transform(train.copy())
    #print(train_scaled)
    #print(train)



    if is_realtime_set:

        realtime_set = create_set_dataframe(df_last.iloc[:, col_set].values, len(df_last.iloc[:, col_set].values), 1, windowX, 0, colmns, bors, is_realtime_set)
        v_min = realtime_set.min(axis=(0, 1), keepdims=True)
        v_max = realtime_set.max(axis=(0, 1), keepdims=True)
        realtime_set = (realtime_set - v_min)/(v_max - v_min)

        return realtime_set, windowX, windowY, v_min, v_max


    else:
        X_train, y_train = create_set_dataframe(train, train_sep, 4, windowX, windowY, colmns, bors)


        v_min = X_train.min(axis=(0, 1), keepdims=True)
        v_max = X_train.max(axis=(0, 1), keepdims=True)
        X_train = (X_train - v_min)/(v_max - v_min)
        if(bors != 'Y'):
            y_train = (y_train - v_min[:,:,0])/(v_max[:,:,0] - v_min[:,:,0])



        #test
        df_volume = np.vstack((train, test))
        inputs = df_volume[df_volume.shape[0] - test.shape[0] - windowX:] #берем с конца элементов в количестве тестовых + окно
        #inputs = inputs.reshape(-1,len(col_set)) #фактически просто проверка размерности
        #inputs = sc.transform(inputs) #типа нормальизованные значения?

        X_test, y_test = create_set_dataframe(inputs, windowX+test.shape[0], 1, windowX, windowY, colmns, bors)
        X_test = (X_test - v_min)/(v_max - v_min)
        if(bors != 'Y'):
            y_test = (y_test - v_min[:,:,0])/(v_max[:,:,0] - v_min[:,:,0])


        print('train', X_train.shape, ' ', y_train.shape) 
        print('test', X_test.shape, ' ', y_test.shape) 
        return X_train, y_train, X_test, y_test, windowX, windowY, v_min, v_max


# %%

'''
print(df)

'''

# %% ###################################################################################################################################
def create_model(X_train, col_set, windowY):
    batch_size = 25
    predict_batch_size = 25


    def model_many_to_one():
        # Initializing the Recurrent Neural Network
        model = Sequential()
        #Adding the first LSTM layer with a sigmoid activation function and some Dropout regularization
        #Units - dimensionality of the output space

        model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], len(col_set))))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50))
        model.add(Dropout(0.2))

        # Adding the output layer
        model.add(Dense(units = 1))
        model.summary()
        return model, model_type

    def model_many_to_one_DG():
        # Initializing the Recurrent Neural Network
        model = Sequential()
        #Adding the first LSTM layer with a sigmoid activation function and some Dropout regularization
        #Units - dimensionality of the output space

        model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], len(col_set))))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50))
        model.add(Dropout(0.2))

        # Adding the output layer
        model.add(Dense(units = 1, activation='sigmoid'))
        model.summary()
        return model, model_type

    def model_simple_seq2seq():
        #тут судя по всему последний юнит предсказывает вектор, а не каждый из windowX = 60, 
        #иначе скорее всего выход был бы тоже windowX = 60
        global model_type
        model_type = 'model_simple_seq2seq'
        model = Sequential()

        model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], len(col_set))))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50))
        model.add(Dropout(0.2))

        # Adding the output layer
        model.add(Dense(windowY))
        model.add(Activation('linear'))
        model.summary()
        
        return model, model_type


    def model_encoder_decoder():
        model_type = 'model_encoder_decoder'
        encoder_inputs = Input(shape=(X_train.shape[1], len(col_set)))
        
        encoder_l1 = LSTM(100, return_sequences = True, return_state=True, dropout=0.0)
        encoder_outputs1 = encoder_l1(encoder_inputs)
        encoder_states1 = encoder_outputs1[1:]
        
        encoder_l2 = LSTM(100, return_sequences = True, return_state=True, dropout=0.0)
        encoder_outputs2 = encoder_l2(encoder_outputs1)
        encoder_states2 = encoder_outputs2[1:]
        
        encoder_l3 = LSTM(100, return_sequences = True , return_state=True, dropout=0.0)
        encoder_outputs3 = encoder_l3(encoder_outputs2)
        encoder_states3 = encoder_outputs3[1:]
        
        encoder_l4 = LSTM(100, return_state=True, dropout=0.0)
        encoder_outputs4 = encoder_l4(encoder_outputs3[0])
        encoder_states4 = encoder_outputs4[1:]
        
        #
        decoder_inputs = RepeatVector(windowY)(encoder_outputs4[0])
        #
        decoder_l1 = LSTM(100, return_sequences=True, dropout=0.0)(decoder_inputs,initial_state = encoder_states1)
        decoder_l2 = LSTM(100, return_sequences=True, dropout=0.0)(decoder_l1,initial_state = encoder_states2)
        decoder_l3 = LSTM(100, return_sequences=True, dropout=0.0)(decoder_l2,initial_state = encoder_states3)   
        decoder_l4 = LSTM(100, return_sequences=True, dropout=0.0)(decoder_l3,initial_state = encoder_states4)   
        
        dropout = Dropout(rate=0.0)
        dropout_outputs = dropout(decoder_l4)
        
        decoder_outputs2 = TimeDistributed(Dense(1, activation='sigmoid'))(dropout_outputs) #Dense(1) количество предсказываемых параметров
        
        #
        model_e2d2 = tensorflow.keras.models.Model(encoder_inputs,decoder_outputs2)
        #
        model_e2d2.summary()
        
        return model_e2d2, model_type


    model, model_type = model_encoder_decoder()
    print('---------------', model_type, '------------')
    return model, batch_size, predict_batch_size, model_type


# %% ###################################################################################################################################

def train_model(model, bors, X_train, y_train, X_test, y_test, batch_size):
    NAME = "Test-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))


    if(bors != 'Y'):
        model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mean_absolute_error'])
    else:
        print('------------bors------------')
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['mean_absolute_error', 'accuracy'])
    # optimizer = MadGrad(lr=0.01)
    '''for i in range(100):
        print(i)
        model.fit(X_train, y_train, epochs = 1, batch_size = batch_size, callbacks=[tensorboard], shuffle = False);
        model.reset_states()'''
    #, validation_data=(X_test,y_test)
    history = model.fit(X_train, y_train, epochs = 100, validation_data=(X_test,y_test), batch_size = batch_size, callbacks=[tensorboard]);
    return model, history, NAME



# %% ###################################################################################################################################
def plot_loss_graph(history, bors):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.show()
    if(bors == 'Y'):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])



def save_model(model):
    #model.save('saved_models/{}'.format(NAME))
    model.save('saved_models/перрвый_вырос_упал')

#model = load_model('saved_models/Test-1626463615')
#save_model(model)



# %% ###################################################################################################################################
def predict(model, v_min, v_max, X_test, predict_batch_size, col_set, model_type):
    #print(X_test[0].reshape(1, 60, 2))
    prediction = model.predict(X_test, batch_size = predict_batch_size)

    '''
    print(test[:,0])
    plt.plot((test[:,colmns['open'].value])[2600:2700])
    plt.show()
    print(predict[2600:2700])
    plt.plot(predict[2600:2700, 0])
    '''
    

    if model_type == 'model_encoder_decoder':
        prediction = prediction[:,:,0]
        
    '''
    if model_type == 'model_encoder_decoder' or model_type == 'model_simple_seq2seq':
        predict0 =[]
        for i in predict:
            predict_dataset_like = np.zeros(shape=(predict.shape[1], len(col_set)) )
            predict_dataset_like[:,0] = i
            predict0.append(sc.inverse_transform(predict_dataset_like)[:,0])

        predict_dataset_like = np.zeros(shape=(len(predict), len(col_set)) )
        predict_dataset_like[:,0] = predict[:,0]
        predict = sc.inverse_transform(predict_dataset_like)[:,0]
    else:
        #преобразование для multi-input
        predict_dataset_like = np.zeros(shape=(len(predict), len(col_set)) ) # create empty table with X fields
        predict_dataset_like[:,0] = predict[:,0] # put the predicted values in the right field
        predict = sc.inverse_transform(predict_dataset_like)[:,0] # inverse transform and then select the right field
    '''

    prediction0 =[]
    #TODO переделать под ручную нормализацию, много лишних преобразований
    if model_type == 'model_encoder_decoder' or model_type == 'model_simple_seq2seq':
        print('encoder_decoder или simple_seq2seq')
        prediction0 =[]
        for i in prediction:
            predict_dataset_like = np.zeros(shape=(prediction.shape[1], len(col_set)) )
            predict_dataset_like[:,0] = i
            prediction0.append(((predict_dataset_like*(v_max[:,:,0] - v_min[:,:,0]) + v_min[:,:,0]))[:,0])

        predict_dataset_like = np.zeros(shape=(len(prediction), len(col_set)) )
        predict_dataset_like[:,0] = prediction[:,0]
        prediction = (prediction*(v_max[:,:,0] - v_min[:,:,0]) + v_min[:,:,0])[:,0]
    else:
        print('multi-input преобразование')
        #преобразование для multi-input
        predict_dataset_like = np.zeros(shape=(len(prediction), len(col_set)) ) # create empty table with X fields
        predict_dataset_like[:,0] = prediction[:,0] # put the predicted values in the right field
        prediction = (prediction*(v_max[:,:,0] - v_min[:,:,0]) + v_min[:,:,0])[:,0] # inverse transform and then select the right field
        
        
        
    #print(predict_dataset_like.shape)
    #print(predict0)
    print(prediction.shape)
    print(len(prediction0))
    #predict = sc.inverse_transform(predict)[:, [2]]
    return prediction, prediction0



# %% ###################################################################################################################################
def plot_metrics(prediction, test, windowY):
    print(prediction.shape)
    if test.shape[1] != 1:
        if windowY == 1:
            diff = prediction - test[:,0]
        else:
            diff = prediction - test[:-windowY+1,0]
    else:
        diff = prediction - test
        
    #print(diff)

    #scores = model.evaluate(X_train, y_train, batch_size = predict_batch_size)
    #print('Точность на тестовых данных: %.2f%%' % (scores[1] * 100))
    #print('F1 на тестовых данных: %.2f%%' % (scores[2] * 100))


    print("MSE:", np.mean(diff**2))
    print("MAE:", np.mean(abs(diff)))
    print("RMSE:", np.sqrt(np.mean(diff**2)))
    return 0


# %% ###################################################################################################################################
def plot_predict_vs_real(prediction, prediction0, train, test, df, model_type, windowX, windowY):
    '''plt.figure(figsize=(20,7))
    plt.plot(df['time'][-predict.shape[0]-500:], df_volume[-predict.shape[0]-500:, 0], color = 'red', label = 'Real BTC Price')
    plt.plot(df['time'][-predict.shape[0]:], predict, color = 'blue', label = 'Predicted BTC Price')

    #plt.xticks(np.arange(100,df[1800:].shape[0],200))
    plt.title('BTC Stock Price Prediction')
    plt.xlabel('time')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()'''

    df_volume = np.vstack((train, test))

    plt.figure(figsize=(20,7))
    if model_type == 'model_encoder_decoder' or model_type == 'model_simple_seq2seq':
        plt.plot(df['open'].index[-prediction.shape[0]-100:], df_volume[-prediction.shape[0]-100:, 0], color = 'red', marker='.', linestyle='', label = 'Real BTC Price')
        plt.plot(df['open'].index[-prediction.shape[0]-windowY+1:-windowY+1], prediction, color = 'blue', marker='.', linestyle='', label = 'Predicted BTC Price')
        for i, x in enumerate(prediction0):
            a = -prediction.shape[0]+i + 1
            a = a if a != 0 else df['open'].shape[0] #TODO как сделать нормально?
            if i == 270:
                plt.plot(df['open'].index[a-windowY:a], x, color = 'green', marker='.', linestyle='-')
    else:
        plt.plot(df['open'].index[-prediction.shape[0]-100:], df_volume[-prediction.shape[0]-100:, 0], color = 'red', marker='.', linestyle='', label = 'Real BTC Price')
        plt.plot(df['open'].index[-prediction.shape[0]:], prediction, color = 'blue', marker='.', linestyle='', label = 'Predicted BTC Price')


    #plt.xticks(np.arange(100,df[1800:].shape[0],200))
    plt.title('BTC Stock Price Prediction')
    plt.xlabel('time')
    plt.ylabel('Price ($)')
    plt.legend()

    plt.show()
    return 0



# %% ###################################################################################################################################
def set_tensorboard():
    import tensorflow.keras.utils
    import pydot
    import pydotplus
    from pydotplus import graphviz

    #tensorflow.keras.utils.plot_model(model)

    #%load_ext tensorboard
    #%tensorboard --logdir logs --host localhost
    return 0

# %% ###################################################################################################################################


# %% ###################################################################################################################################
'''
#работает кода для предсказания используем только цену

pred_ = predict[-1].copy()
prediction_full = []
#window = 60
df_copy = df.iloc[:, 1:2][1:].values

future_times = 50


for j in range(future_times):
    df_ = np.vstack((df_copy, pred_))
    train_ = df_[:train_sep]
    test_ = df_[train_sep:test_sep]
    
    df_volume_ = np.vstack((train_, test_))

    inputs_ = df_volume_[df_volume_.shape[0] - test_.shape[0] - windowX:]
    inputs_ = inputs_.reshape(-1,1)
    inputs_ = sc.transform(inputs_)

    X_test_2 = []

    for k in range(windowX, num_2):
        X_test_3 = np.reshape(inputs_[k-windowX:k, 0], (windowX, 1))
        X_test_2.append(X_test_3)

    X_test_ = np.stack(X_test_2) #[:25] временный костыль для stateful ломает вывод
    predict_ = model.predict(X_test_, batch_size = predict_batch_size)
    pred_ = sc.inverse_transform(predict_)
    prediction_full.append(pred_[-1][0])
    df_copy = df_[j:]
    
    
prediction_full_new = np.vstack((predict, np.array(prediction_full).reshape(-1,1)))

'''
# %% ###################################################################################################################################

'''
df_date = df[['time']]

time_interval = df_date['time'].iloc[-1] - df_date['time'].iloc[-2]

for m in range(future_times):
    df_date_add = df_date['time'].iloc[-1] + time_interval
    df_date_add = pd.DataFrame([df_date_add], columns=['time'])
    df_date = df_date.append(df_date_add)

df_date = df_date.reset_index(drop=True)
'''
# %% ###################################################################################################################################

'''
#plt.figure(figsize=(20,7))
#plt.plot(df['time'][-predict.shape[0]-10:], df_volume[-predict.shape[0]-10:], color = 'red', linestyle='', marker='o', label = 'Real BTC Price')
#plt.plot(df_date['time'][-prediction_full_new.shape[0]:], prediction_full_new, linestyle='', marker='o', color = 'blue', label = 'Predicted BTC Price')


#####
df_volume2 = np.vstack((train, test, real_test))
predict2 = np.vstack((predict, real_test))

plt.figure(figsize=(20,7))
plt.plot(df['time'][-predict.shape[0]-500:-real_test_size], df_volume[-predict.shape[0]-500+real_test_size:], color = 'red', label = 'Real BTC Price')
plt.plot(df['time'][-predict.shape[0]-real_test_size:-real_test_size], predict2[:-real_test_size], color = 'blue', label = 'Predicted BTC Price')
plt.plot(df['time'][-real_test_size:], df_volume2[-real_test_size:], color = 'yellow', label = 'Real BTC Price')
#####


plt.figure(figsize=(20,7))
plt.plot(df_date['open'].index[-prediction_full_new.shape[0]:], prediction_full_new, linestyle='', marker='.', color = 'blue', label = 'Predicted BTC Price')

plt.plot(df['open'].index[-prediction_full_new.shape[0]-30:],  df['open'][-prediction_full_new.shape[0]-30:], color = 'red', linestyle='', marker='.', label = 'Real BTC Price')
plt.plot(df_o['open'].index[-real_test_size:], df_o['open'].values[-real_test_size:], color = 'green', marker='.', linestyle='', label = 'Real BTC Price')



#plt.xticks(np.arange(100,df[1800:].shape[0],200))
plt.title('BTC Price Prediction')
plt.xlabel('time')
plt.ylabel('Price ($)')
plt.legend()
plt.show()


'''
# %% ###################################################################################################################################


# %% ###################################################################################################################################


# %% ###################################################################################################################################


# %% ###################################################################################################################################


# %%

