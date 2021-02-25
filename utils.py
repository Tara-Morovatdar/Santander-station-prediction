import os
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error as MAE
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller
from sklearn import preprocessing

def clean_df(temp_df,station_id,start_date,end_date,freq):  
    """This function does couple of cleaning steps:
        1-makes the data frame a datetime df
        2- makes sure that the frequency of the data is 2 mins
        3- takes the part of the data that we need acording to the start and end date
        4- fills the missing values in betweeb with the value of previous row"""
    df_temp=temp_df.copy()
    df_temp['timestamp']=df_temp['timestamp'].values.astype('<M8[m]')
    df_temp.drop_duplicates('timestamp',inplace=True)
    
    # set the frequency of values to 2 minutes so we dont have any missing observation
    df_temp = df_temp.set_index('timestamp')
    #df_temp.loc['2018-12-31 23:58:00']=np.nan
    df_temp=df_temp.asfreq(freq='2Min')
    df_temp=df_temp[['spaces']]
    ##used to distinguish trucks
    #df_temp['spaces']=median_filter(df_temp, varname='spaces', window=10)
    #df_temp['spaces']=df_temp['spaces'].rolling(window=2).mean()
    df_temp=df_temp.fillna(method='ffill')
    df_temp=df_temp.fillna(method='bfill')
    df_temp=df_temp.resample(str(freq)+'min').mean()
    df_temp=df_temp[start_date:end_date]  
    df_temp=df_temp.rename(columns={'spaces':'spaces_'+str(station_id)})
    return df_temp



def normalize(temp_df,total_dock):
    '''Normalize(z score normalization mean is around zero and std is 1)'''
    df_temp=temp_df.copy()
    max_value=total_dock
    min_value=0

    df_temp.iloc[:,0] = (df_temp.iloc[:,0] - min_value) / (max_value-min_value)

    return df_temp

def inverse_normalize(temp_df,total_dock):
    df_temp=temp_df.copy()
    df_temp.iloc[:,0] = df_temp.iloc[:,0]*total_dock
    return df_temp

def anomaly_detection(df_temp,freq,N=3):
    '''anomaly detection using z-score
    residual should have normal distribution 
    so anything above N std is anomaly
    we use seasonal decompose to get residuals
    '''
    spaces_series_np=df_temp.iloc[:, 0].to_numpy()
    result = seasonal_decompose(spaces_series_np, model='additive', extrapolate_trend='freq',period=int(10080/freq))
    seasonal , trend, resid = result.seasonal , result.trend, result.resid
    resid_mu = resid.mean()
    resid_dev = resid.std()
    #anomalies
    lower = resid_mu - N*resid_dev
    upper = resid_mu + N*resid_dev
    anomalies = df_temp[(resid < lower) | (resid > upper)]
    return anomalies
    
def anomaly_removal(anomalies,df_temp):
    '''This function gets a dataframe of anomalies andsubstitude the anomalies
    with the average value of the same day and time along our data
    '''
    df_avgs=df_temp.groupby([df_temp.index.dayofweek,df_temp.index.time]).mean().rename_axis(['day','time']).reset_index()
    for i in range(0,len(anomalies)):
          df_temp.loc[anomalies.index[i]]=df_avgs[(df_avgs.day==anomalies.index[i].weekday()) &
                                                       (df_avgs.time==anomalies.index[i].time())].iloc[0,2]
    
    return df_temp


def create_error_report(df_results,df_test,forecast_steps,d):
    eval_report=[]
    for name, column in (df_test.iteritems()):
        
        result_dict=forecast_accuracy(df_results[name+'_forecast_'+str(d)].values, df_test[name].values,forecast_steps)
        #result_dict['Station_id']=int(name.split('_')[1])
        result_dict['Station_id']=int(name.split('_')[1])
        result_dict['Month']=df_test.index[1].month
        eval_report.append(result_dict)

    return eval_report


def forecast_accuracy(forecast, actual,forecast_steps):
    """Perform evaluation metrics"""
    forecast=forecast[forecast_steps-1::forecast_steps]
    actual=actual[forecast_steps-1::forecast_steps]

    mae = np.mean(np.abs(np.array(forecast) - np.array(actual)))    # MAE

    rmse = np.mean((np.array(forecast) - np.array(actual))**2)**.5  # RMSE
    
    #rrmse=  np.mean((forecast - actual)**2)**.5 / np.mean(forecast) *100 #RRMSE


    return round(mae,4),round(rmse,4)

def forecast_truncate(temp_df,total_docks):
    """If the predictions are out of the bounderies(les than zero or greater than the capacity) we cut off"""
    df_temp=temp_df.copy()
    for name, column in (df_temp.iteritems()):
        
        #total_docks=df[df['operator_id']==int(name.split('_')[1])]['total_docks'].values[0]
        for i in range(0,len(column)):
            if column[i]<0:
                column[i]=0

            elif column[i]>total_docks:
                column[i]=total_docks
    return df_temp


def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    
#     s1=series.iloc[0:int(len(series)/2)]
#     s2=series.iloc[int(len(series)/2):]
#     r = adfuller(s1, autolag='AIC')
#     output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
#     p_value = output['pvalue'] 
    r1 = adfuller(series, autolag='AIC')
    output1 = {'test_statistic':round(r1[0], 4), 'pvalue':round(r1[1], 4), 'n_lags':round(r1[2], 4), 'n_obs':r1[3]}
    p_value1 = output1['pvalue'] 
    
    if (p_value1 <= signif):

        return True
        
    else:

        return False

def make_Stationary(temp_df):
    """Take difference until the data is stationary"""
    df_temp=temp_df.copy()
    diff_count=0
    stationarity=[]
    while diff_count<3:
        for name, column in df_temp.iteritems():
            
            s=adfuller_test(column, name=column.name)
            stationarity.append(s)
            #print(stationarity)
        if all(stationarity):
           
            break
        else:
            diff_count+=1
            df_temp = df_temp.diff().fillna(0)
            stationarity=[]
        
    return df_temp,diff_count   



def invert_transformation_forecast(df1, df_forecast,test_size, diff_count):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df1.columns
    for col in columns:        
        # Roll back 2nd Diff
        if diff_count==2:
            df_fc[str(col)+'_1'] = (df1[col].iloc[-test_size-1]-df1[col].iloc[-test_size-2]) + df_fc[str(col)+'_forecast_'+'_2'].cumsum()
            # Roll back 1st Diff
            df_fc[str(col)+'_forecast_0'] = df1[col].iloc[-test_size-1] + df_fc[str(col)+'_1'].cumsum()
            df_fc=df_fc.drop(str(col)+'_forecast'+'_2',axis=1)
            df_fc=df_fc.drop(str(col)+'_forecast'+'_1',axis=1)
        if diff_count==1:
            df_fc[str(col)+'_forecast_0'] = df1[col].iloc[-test_size-1] + df_fc[str(col)+'_forecast'+'_1'].cumsum()
            df_fc=df_fc.drop(str(col)+'_forecast'+'_1',axis=1)
      
    return df_fc

def invert_transformation_test(df1, df_forecast,test_size, diff_count):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df1.columns
    for col in columns:        
        # Roll back 2nd Diff
        if diff_count==2:
            df_fc[str(col)+'_1'] = (df1[col].iloc[-test_size-1]-df1[col].iloc[-test_size-2]) + df_fc[str(col)].cumsum()
            # Roll back 1st Diff
            df_fc[str(col)+'_forecast_0'] = df1[col].iloc[-test_size-1] + df_fc[str(col)+'_1'].cumsum()
            df_fc=df_fc.drop(str(col)+'_forecast'+'_2',axis=1)
            df_fc=df_fc.drop(str(col)+'_forecast'+'_1',axis=1)
        if diff_count==1:
            df_fc[str(col)+'_forecast_0'] = df1[col].iloc[-test_size-1] + df_fc[str(col)].cumsum()
            df_fc=df_fc.drop(str(col),axis=1)
      
    return df_fc
def clean_exg_df(df_temp,start_date,end_date,freq):
    
    temp_df=df_temp.copy()
    temp_df['timestamp']=temp_df['timestamp'].values.astype('<M8[m]')
    temp_df.drop_duplicates('timestamp',inplace=True)
    exg_features=['timestamp','temp', 'humidity','hour', 'Rain', 'is_rushhour','is_day','is_non_workday']
    exg_features=['timestamp','temp']
    #exg_features=['timestamp','temp','pressure', 'wind_speed','humidity', 'hour', 'is_rushhour', 'is_day']
#     exg_features=['timestamp','temp', 'pressure', 'rain_1h', 'humidity', 'clouds_all', 'Clear',
#        'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain', 'is_weekend',
#        'is_holiday', 'is_non_workday', 'hour', 'is_rushhour', 'is_day']
    temp_df=temp_df[exg_features]
    X = temp_df.drop(['timestamp'], axis=1)
    y = temp_df[['timestamp']]
    scaler_x = preprocessing.MinMaxScaler()
    X =  pd.DataFrame(scaler_x.fit_transform(X), columns = X.columns)
    X['timestamp']=y
    X.timestamp = pd.DatetimeIndex(X.timestamp.values)
    temp_df=X.copy()
    temp_df['timestamp']=temp_df['timestamp'].values.astype('<M8[m]')
    temp_df.drop_duplicates('timestamp',inplace=True)
    #temp_df=temp_df.resample(str(freq)+'min').mean()
    temp_df=temp_df.set_index('timestamp')
    temp_df=temp_df.asfreq(freq=str(freq)+'min')
    temp_df=temp_df[start_date:end_date] 
    temp_df=temp_df.fillna(method='ffill')
    return temp_df
    

def plot_series(series):
    plt.figure(figsize=(15,6))
    plt.plot(series)
    plt.ylabel('Number of Slots', fontsize=16)
    plt.xlabel('Date', fontsize=16)
