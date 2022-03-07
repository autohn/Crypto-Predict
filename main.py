
# %%

'''
import perdict_VSCode as q
'''

import importlib
import perdict_VSCode
importlib.reload(perdict_VSCode)
from perdict_VSCode import *


# %%
crypto_name = 'Bitcoin'

file_name = 'btcusd_my.csv'

is_realtime_set = 'Y'

df_o = load_dataset(file_name, '1T')
df_last = load_dataset(file_name, '1T', is_realtime_set)


def mass_enrich(df_o, crypto_name):
    df_o = mass_load_dataset(df_o)

    df_o = fill_nan(df_o)

    #get_trends(crypto_name, 2014, 10, 2021, 6, '')

    df_o = load_trends(crypto_name, 'google_tranding_all_d_2014_2021.csv', df_o)

    #df_o = data_to_sin(df_o)
    return df_o

#mass_enrich(df_o, crypto_name)

print_dataset(df_o, crypto_name)



# %%
#df, train, test, real_test, test_size, real_test_size, train_sep, test_sep, col_set = old_data_partitioning(df_o)
#X_train, y_train, sc = old_train_create(train)
#X_test, y_test = old_test_create(train, test)

col_set, colmns = get_col_set(df_o)

df, train, test, train_sep = data_partitioning(df_o, crypto_name, col_set)


bors =  'N'
X_train, y_train, X_test, y_test, windowX, windowY, v_min, v_max = create_dataset(train, test, train_sep, col_set, colmns, bors)


#realtime_set, windowX, windowY, v_min, v_max = create_dataset(train, test, train_sep, col_set, colmns, bors, is_realtime_set, df_last)




# %%


model, batch_size, predict_batch_size, model_type = create_model(X_train, col_set, windowY)

# %%

model, history, NAME = train_model(model, bors, X_train, y_train, X_test, y_test, batch_size)

plot_loss_graph(history, bors)

#model = load_model('saved_models/перрвый_вырос_упал')
#save_model(model)



# %%
prediction, prediction0 = predict(model, v_min, v_max, X_test, predict_batch_size, col_set, model_type)

plot_metrics(prediction, test, windowY)

plot_predict_vs_real(prediction, prediction0, train, test, df, model_type, windowX, windowY)

# %%
print(df_o.columns.tolist().index(crypto_name))

df_o[["Open", "Close", "Low", "High"]].corr()
# %%


# %%
