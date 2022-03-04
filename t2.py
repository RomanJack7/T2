import warnings;
warnings.simplefilter('ignore')
import pandas as pd
import streamlit as st
import numpy as np
import warnings
import itertools
import matplotlib
import matplotlib.pyplot as plt
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.stattools import adfuller
import plotly.figure_factory as ff

st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings("ignore")

plt.style.use('seaborn-bright')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

import pandas as pd
# Import the data
df = pd.read_csv("Blog_Orders.csv")
df['Date'] = pd.to_datetime(df['Date'])
# Set the date as index 
df = df.set_index('Date')
# Select the proper time period for weekly aggreagation
df = df['2013-01-01':'2017-12-31'].resample('W').sum()
df.head()  
#st.sidebar.selectbox("Choose the forecasting method to continue", ['Simple Exponential Smoothing','HOLTs Forecasting Method', 'HOLT-WINTERs Forecasting Method'])
rad = st.sidebar.radio("Navigation - Choose a forecasting method to continue",["Visualising Your Data","Simple Exponential Smoothing","HOLTs Forecasting Method","HOLT-WINTERs Forecasting Method"])

if rad == "Visualising Your Data":
    header = st.container() 
    from PIL import Image
    
    with header:
        image = Image.open('cslp_logo_1.png')
        st.image(image)
        st.text('INDU 6990 - INDUSTRIAL ENGINEERING CAPSTONE')
        st.text('GROUP 16')
        st.title('FORECASTING')    
else:
    pass
        
###############################################################################

import warnings
import matplotlib.pyplot as plt
# Orders Graph
y = df['Orders']
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(y,marker='.', linestyle='-', linewidth=0.5, label='Weekly')
ax.plot(y.resample('M').mean(),marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample')
ax.set_ylabel('Orders')
ax.legend();
orders = st.container()

if rad == "Visualising Your Data":
    with orders:
        st.header('Graph of your Orders')
        st.pyplot(fig)

###############################################################################

#Seasonal Decomposed Graph
import statsmodels.api as sm

# graphs to show seasonal_decompose
def seasonal_decompose (y):
    decomposition = sm.tsa.seasonal_decompose(y, model='additive',extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()
decompose = st.container()
seasonal_decompose(y) 

if rad == "Visualising Your Data":
    with decompose:
        st.header('Here you can see individual elements of your orders data. TREND AND SEASONALITY')
        st.pyplot(y.empty)

###############################################################################

### plot for Rolling Statistic for testing Stationarity
def test_stationarity(timeseries, title):
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=12).mean() 
    rolstd = pd.Series(timeseries).rolling(window=12).std()
    
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(timeseries, label= title)
    ax.plot(rolmean, label='rolling mean');
    ax.plot(rolstd, label='rolling std (x10)');
    ax.legend()

pd.options.display.float_format = '{:.8f}'.format
test_stationarity(y,'raw data')

# Augmented Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller

def ADF_test(timeseries, dataDesc):
    print(' > Is the {} stationary ?'.format(dataDesc))
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    print('Test statistic = {:.3f}'.format(dftest[0]))
    print('P-value = {:.3f}'.format(dftest[1]))
    print('Critical values :')
    for k, v in dftest[4].items():
        print('\t{}: {} - The data is {} stationary with {}% confidence'.format(k, v, 'not' if v<dftest[0] else '', 100-int(k[:-1])))

ADF_test(y,'raw data')

# Detrending
y_detrend =  (y - y.rolling(window=12).mean())/y.rolling(window=12).std()

test_stationarity(y_detrend,'de-trended data')
ADF_test(y_detrend,'de-trended data')

# Differencing
y_12lag =  y - y.shift(12)

test_stationarity(y_12lag,'12 lag differenced data')
ADF_test(y_12lag,'12 lag differenced data')

# Detrending + Differencing

y_12lag_detrend =  y_detrend - y_detrend.shift(12)

test_stationarity(y_12lag_detrend,'12 lag differenced de-trended data')
ADF_test(y_12lag_detrend,'12 lag differenced de-trended data')

y_to_train = y[:'2017-07-02'] # dataset to train
y_to_val = y['2016-07-03':] # last X months for test  
predict_date = len(y) - len(y[:'2016-07-03']) # the number of data points for the test set

###############################################################################

if rad == "Simple Exponential Smoothing":   
       
    import numpy as np
    from statsmodels.tsa.api import SimpleExpSmoothing

    def ses(y, y_to_train,y_to_test,smoothing_level,predict_date):
        y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
        
        fit1 = SimpleExpSmoothing(y_to_train).fit(smoothing_level=smoothing_level,optimized=False)
        fcast1 = fit1.forecast(predict_date).rename(r'$\alpha={}$'.format(smoothing_level))
        # specific smoothing level
        fcast1.plot(marker='o', color='blue', legend=True)
        fit1.fittedvalues.plot(marker='o',  color='blue')
        mse1 = ((fcast1 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of our forecasts with smoothing level of {} is {}'.format(smoothing_level,round(np.sqrt(mse1), 2)))
        
        ## auto optimization
        fit2 = SimpleExpSmoothing(y_to_train).fit()
        fcast2 = fit2.forecast(predict_date).rename(r'$\alpha=%s$'%fit2.model.params['smoothing_level'])
        # plot
        fcast2.plot(marker='o', color='green', legend=True)
        fit2.fittedvalues.plot(marker='o', color='green')
        
        mse2 = ((fcast2 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of our forecasts with auto optimization is {}'.format(round(np.sqrt(mse2), 2)))
        
        plt.show()
        
    sesgraph = ses(y, y_to_train,y_to_val,0.8,predict_date)  


    simple_exponintial_smoothing = st.container()
    with simple_exponintial_smoothing:
        st.header('Simple exponential smoothing')
        st.text('This graph shows forecast using simple exponential smoothing.')
        st.text('NOTE : This is a basic forecasting method and does not include any trend or')
        st.text('seasonality factor')
        st.pyplot(sesgraph)

###############################################################################

if rad == "HOLTs Forecasting Method":
    from statsmodels.tsa.api import Holt
    
    def holt(y,y_to_train,y_to_test,smoothing_level,smoothing_slope, predict_date):
        y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
        
        fit1 = Holt(y_to_train).fit(smoothing_level, smoothing_slope, optimized=False)
        fcast1 = fit1.forecast(predict_date).rename("Holt's linear trend")
        mse1 = ((fcast1 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of Holt''s Linear trend {}'.format(round(np.sqrt(mse1), 2)))
    
        fit2 = Holt(y_to_train, exponential=True).fit(smoothing_level, smoothing_slope, optimized=False)
        fcast2 = fit2.forecast(predict_date).rename("Exponential trend")
        mse2 = ((fcast2 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of Holt''s Exponential trend {}'.format(round(np.sqrt(mse2), 2)))
        
        fit1.fittedvalues.plot(marker="o", color='blue')
        fcast1.plot(color='blue', marker="o", legend=True)
        fit2.fittedvalues.plot(marker="o", color='red')
        fcast2.plot(color='red', marker="o", legend=True)
    
        plt.show()
        
    fig = holt(y, y_to_train,y_to_val,0.6,0.2,predict_date)
    
    holt_graph = st.container()
    
    #if rad == "HOLTs Forecasting Method":
    with holt_graph:
        st.header('HOLTs Forecasting Method')
        st.text('This graph shows forecast using HOLTs Forecasting Method')
        st.text('NOTE : This forecasting method considers trend factor and is more')
        st.text('accurate than Simple exponential Smoothing')
        st.pyplot(fig)
        plt.close(fig)

###############################################################################

elif rad == "HOLT-WINTERs Forecasting Method":
    from statsmodels.tsa.api import ExponentialSmoothing
    
    def holt_win_sea(y,y_to_train,y_to_test,seasonal_type,seasonal_period,predict_date):
        
        y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
        
        if seasonal_type == 'additive':
            fit1 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='add').fit(use_boxcox=True)
            fcast1 = fit1.forecast(predict_date).rename('Additive')
            mse1 = ((fcast1 - y_to_test) ** 2).mean()
            print('The Root Mean Squared Error of additive trend, additive seasonal of '+ 
                  'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse1), 2)))
            
            fit2 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
            fcast2 = fit2.forecast(predict_date).rename('Additive+damped')
            mse2 = ((fcast2 - y_to_test) ** 2).mean()
            print('The Root Mean Squared Error of additive damped trend, additive seasonal of '+ 
                  'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse2), 2)))
            
            fit1.fittedvalues.plot(style='--', color='red')
            fcast1.plot(style='--', marker='o', color='red', legend=True)
            fit2.fittedvalues.plot(style='--', color='green')
            fcast2.plot(style='--', marker='o', color='green', legend=True)
        
        elif seasonal_type == 'multiplicative':  
            fit3 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='mul').fit(use_boxcox=True)
            fcast3 = fit3.forecast(predict_date).rename('Multiplicative')
            mse3 = ((fcast3 - y_to_test) ** 2).mean()
            print('The Root Mean Squared Error of additive trend, multiplicative seasonal of '+ 
                  'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse3), 2)))
            
            fit4 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='mul', damped=True).fit(use_boxcox=True)
            fcast4 = fit4.forecast(predict_date).rename('Multiplicative+damped')
            mse4 = ((fcast3 - y_to_test) ** 2).mean()
            print('The Root Mean Squared Error of additive damped trend, multiplicative seasonal of '+ 
                  'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse4), 2)))
            
            fit3.fittedvalues.plot(style='--', color='red')
            fcast3.plot(style='--', marker='o', color='red', legend=True)
            fit4.fittedvalues.plot(style='--', color='green')
            fcast4.plot(style='--', marker='o', color='green', legend=True)
            
        else:
            print('Wrong Seasonal Type. Please choose between additive and multiplicative')
    
        plt.show()
        
    fig = holt_win_sea(y, y_to_train,y_to_val,'additive',52, predict_date)
    
    holt_winter = st.container()
    
    
    with holt_winter:
        st.header('HOLT-WINTERs Forecasting Method')
        st.text('This graph shows forecast using HOLT-WINTERs Forecasting Method')
        st.text('NOTE : This forecasting method considers trend and seasonality factor')
        st.text('and is more accurate than simple exponential smoothing and HOLTs')
        st.text('forecasting method')
        st.pyplot(fig)
        plt.close(fig)
else:
    pass

###############################################################################



    


    
    


