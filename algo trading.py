import numpy as np , pandas as pd, matplotlib.pyplot as plt
import sklearn as sk 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import yfinance as yf 
import tensorflow as tf 
import pandas_ta as ta , mplfinance as mpl 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Convolution1D, Conv1D, MaxPooling1D, BatchNormalization, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy, categorical_crossentropy
from tensorflow.keras.metrics import MeanAbsoluteError
from sklearn.metrics import r2_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

#Getting the data
ticker = "GOOG"
pricedata = yf.download(ticker, start='2022-01-01', auto_adjust=True)
plt.plot(pricedata['Close'])

# Stock Information
stock = yf.Ticker(ticker)
info = stock.info
sector =info.get('sector')
industry = info.get('industry')
print(industry)
print(sector)
pe = info.get('forwardPE')
revenue = info.get('totalRevenue', 'N/A')
net_profit = info.get('netIncomeToCommon', 'N/A')


# RSI
pricedata['RSI'] = ta.rsi(pricedata['Close'], length= 14)
pricedata.dropna(subset=['RSI'], inplace = True)

#OBV
pricedata['OBV'] = ta.obv(pricedata['Close'], pricedata['Volume'])


'''
# Stochastic Oscillator (%K and %D)
stoch = ta.stoch(pricedata['High'], pricedata['Low'], pricedata['Close'], 14, 3, 1)
pricedata["stoch_K"] = stoch['STOCHk_14_3_1']
pricedata["stoch_D"] = stoch['STOCHd_14_3_1']'''

#Bollinger Bands
bb = ta.bbands(pricedata['Close'], length = 20, std =2, ddof =0, mamode='ema',talib=False)
pricedata= pricedata.join(bb)


#Fibonacci wma
pricedata['FWMA'] = ta.fwma(pricedata['Close'], length = 15)

#ADX
adx_calc = ta.adx(pricedata['High'], pricedata['Low'], pricedata["Close"], length = 15)

pricedata['ADX'] = adx_calc['ADX_15']
pricedata['+DI'] = adx_calc['DMP_15']
pricedata['-DI'] = adx_calc['DMN_15']



#Awesome Oscillator
pricedata['AO'] = ta.ao(pricedata['High'], pricedata["Low"], fast =5, slow = 34)


#Even Better Sinewave
ebsw = ta.ebsw(pricedata['Close'], length=15)
pricedata['EBSW'] = ebsw


#MACD
macd = ta.macd(pricedata['Close'], fast= 12, slow = 26, signal = 9)
pricedata['MACD'] = macd['MACD_12_26_9']
pricedata['MACD High'] = macd['MACDh_12_26_9']
pricedata['MACD Low'] = macd['MACDs_12_26_9']


#ATR
atr = ta.atr(pricedata['High'], pricedata['Low'], pricedata["Close"])
pricedata['ATR'] = atr



#Window shaping
window = 34
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'OBV','BBL_20_2.0','BBM_20_2.0','BBU_20_2.0','BBB_20_2.0','BBP_20_2.0','FWMA','ADX','+DI','-DI','AO','EBSW','MACD', 'MACD High','MACD Low','ATR']



def create_timesteps(pricedata, window):
    data =[]
    
    for i in range(window, len(pricedata)):
        timestep = pricedata[features].iloc[i-window:i].values
        data.append(timestep)
    return np.array(data)

# Creating labels for buy, sell and hold
closing_price = pricedata['Close']
returns = closing_price.pct_change().fillna(0)
threshold = 0.01
y = np.zeros(len(returns), dtype=int)

y[returns>threshold] =1
y[returns<-threshold] =2
y[np.abs(returns)<= threshold] = 3

y = y.reshape(-1,1)
encoder = OneHotEncoder(sparse_output=False)
y= encoder.fit_transform(y)

print(y.shape)
print(type(y))
print(y.dtype)


# training variables
X= create_timesteps(pricedata, window)  

X_reshaped= X.reshape(X.shape[0], -1)


imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_reshaped)

X = X_imputed.reshape(X.shape)

if len(X) < len(y):
    y = y[:len(X)]

train_size = int(0.7 * len(X))  # 70% for training
X_train, t_data = X[:train_size], X[train_size:]
y_train, temp = y[:train_size], y[train_size:]

val_size = int(0.5 * len(t_data))  # 50% of the remaining 30% for validation
X_val, X_test = t_data[:val_size], t_data[val_size:]
y_val, y_test = temp[:val_size], temp[val_size:]

# Step 2: Verify shapes before scaling
print("X_train shape:", X_train.shape)  
print("X_val shape:", X_val.shape)      
print("X_test shape:", X_test.shape)    



# Step 3: Scaling (Fit on X_train only)
scaler =    MinMaxScaler()

# Reshaping to 2D for scaling (samples, features)
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])  
X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)  # Reshape back to (samples_train, 34, 24)

# Apply scaling to validation and test sets 
X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])  
X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)  # Reshape back to (samples_val, 34, 24)

X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])  
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)  # Reshape back to (samples_test, 34, 24)

# Step 4: Verify shapes after scaling
print("X_train_scaled shape:", X_train_scaled.shape)  
print("X_val_scaled shape:", X_val_scaled.shape)      
print("X_test_scaled shape:", X_test_scaled.shape)    



#Handling class imbalance
class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=np.argmax(y_train, axis=1))
class_weights = dict(enumerate(class_weights))


#Neural Network Architecture
model = Sequential([
    Conv1D(kernel_size= 3, filters = 64, input_shape=([34,22]), activation='relu', padding='causal'),
    Bidirectional(GRU(units =15, activation='relu')),
    Dropout(0.25),
    Dense(units =3, activation='softmax', kernel_regularizer=l2(0.005))
])
model.summary()
model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])



fitting = model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs =100, batch_size=64, class_weight= class_weights)

loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test loss: {loss}")
print(f"Accuracy: {accuracy}")

train_loss = fitting.history['loss']  # Training loss for each epoch
val_loss = fitting.history['val_loss']  # Validation loss for each epoch

# Print the losses:
print(f"Training Loss (J_train) after last epoch: {train_loss[-1]}")
print(f"Validation Loss (J_val) after last epoch: {val_loss[-1]}")

y_pred = model.predict(X_test_scaled)
print("Train class distribution:", Counter(np.argmax(y_train, axis=1)))
print("Test class distribution:", Counter(np.argmax(y_test, axis=1)))

y_test_labels = np.argmax(y_test, axis=1)  
y_pred_labels = np.argmax(y_pred, axis=1)

print(confusion_matrix(y_test_labels, y_pred_labels))#Confusion matrix
print(classification_report(y_test_labels, y_pred_labels))#Classification REport

