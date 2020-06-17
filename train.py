import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import numpy as np
from data_preparation import *
from utils import *
from keras.layers import Bidirectional
from keras.layers.core import Dense, Flatten
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.layers.convolutional import Conv1D
from attention import AttentionLayer
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint


#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
print('loading data...')
data1 = load_csv(r'data-freeway/10105110', 8, "freeway")
data2 = load_csv(r'data-freeway/10105310', 8, "freeway")
data3 = load_csv(r'data-freeway/10105510', 8, "freeway")
data4 = load_csv(r'data-freeway/10108210', 8, "freeway")
data5 = load_csv(r'data-freeway/10106510', 8, "freeway")
data6 = load_csv(r'data-freeway/1095110', 8, "freeway")
data7 = load_csv(r'data-freeway/1095510', 8, "freeway")

# data1 = load_csv(r'data-urban/401190', 5, "urban")
# data2 = load_csv(r'data-urban/401144', 7, "urban")
# data3 = load_csv(r'data-urban/401413', 11, "urban")
# data4 = load_csv(r'data-urban/401911', 8, "urban")
# data5 = load_csv(r'data-urban/401610', 10, "urban")
# data6 = load_csv(r'data-urban/401273', 8, "urban")
# data7 = load_csv(r'data-urban/401137', 8, "urban")

epoch = 50
day = 288
week = 2016
seq_len = 15
#1=5min, 3=15min, 6=30min, 12=60min
pre_len = 12
#data 1-7
pre_sens_num = 1

#train,test
train_data, train_w, train_d, label, test_data, test_w, test_d, test_l, test_med, test_min\
	= generate_data(data1, data2, data3, data4, data5, data6, data7, seq_len, pre_len, pre_sens_num)

train_data = np.reshape(train_data,(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
train_w = np.reshape(train_w,(train_w.shape[0], train_w.shape[1], 1))
train_d = np.reshape(train_d,(train_d.shape[0], train_d.shape[1], 1))

test_data = np.reshape(test_data,(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))
test_d = np.reshape(test_d,(test_d.shape[0], test_d.shape[1], 1))
test_w = np.reshape(test_w,(test_w.shape[0], test_w.shape[1], 1))


# conv-lstm
main_input = Input((15, 7, 1),name='main_input')
con1 = TimeDistributed(Conv1D(filters=15, kernel_size=3, padding='same', activation='relu', strides=1))(main_input)
con2 = TimeDistributed(Conv1D(filters=15, kernel_size=3, padding='same', activation='relu', strides=1))(con1)
#con3 = TimeDistributed(AveragePooling1D(pool_size=2))(con2)
con_fl = TimeDistributed(Flatten())(con2)
con_out = Dense(15)(con_fl)

lstm_out1 = LSTM(15, return_sequences=True)(con_out)
lstm_out2 = LSTM(15, return_sequences=False)(lstm_out1)
lstm_out3 = AttentionLayer()([lstm_out2, con_out])

# Bilstm
auxiliary_input_w = Input((15, 1), name='auxiliary_input_w')
lstm_outw1 = Bidirectional(LSTM(15, return_sequences=True))(auxiliary_input_w)
lstm_outw2 = Bidirectional(LSTM(15, return_sequences=False))(lstm_outw1)

auxiliary_input_d = Input((15, 1), name='auxiliary_input_d')
lstm_outd1 = Bidirectional(LSTM(15, return_sequences=True))(auxiliary_input_d)
lstm_outd2 = Bidirectional(LSTM(15, return_sequences=False))(lstm_outd1)

x = keras.layers.concatenate([lstm_out3, lstm_outw2, lstm_outd2])
x = Dense(20, activation='relu')(x)
x = Dense(10, activation='relu')(x)
main_output = Dense(1, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(0.1, 0.1), name='main_output')(x)
model = Model(inputs = [main_input, auxiliary_input_w, auxiliary_input_d], outputs = main_output)
model.summary()
model.compile(optimizer='adam', loss=my_loss)

#train_save model
filepath = "model/model_{epoch:04d}-{val_loss:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                             mode='min',period=5)
callbacks_list = [checkpoint]

print('Train...')
model.fit([train_data, train_w, train_d], label,
		  batch_size=128, epochs=epoch, validation_split=0.15, verbose=2,
		  class_weight='auto', callbacks=callbacks_list)
model_json = model.to_json()
with open("model/conv_lstm.json", "w") as json_file:
    json_file.write(model_json)
print("Save model to disk")

#load model
# with CustomObjectScope({'AttentionLayer': AttentionLayer}):
# 	json_file = open('model/conv_lstm.json', 'r')
# 	loaded_model_json = json_file.read()
# 	json_file.close()
# 	cnn_lstm_model = model_from_json(loaded_model_json)
# 	cnn_lstm_model.load_weights("model/model_0200-0.0337.h5", 'r')

#Predict the last model
predicted = predict_point_by_point(model, [test_data, test_w, test_d])
p_real = []
l_real = []
row=2016
for i in range(row):
    p_real.append(predicted[i] * test_med + test_min)
    l_real.append(test_l[i] * test_med + test_min)
p_real = np.array(p_real)
l_real = np.array(l_real)

print ("MAE:", MAE(p_real, l_real))
print ("MAPE:", MAPE(p_real, l_real))
print ("RMSE:", RMSE(p_real, l_real))
