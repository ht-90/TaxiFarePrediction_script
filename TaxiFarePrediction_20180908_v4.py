
# coding: utf-8

# Model Devlopment Methodologies
# 
# Data Review
# - Descriptive statistics
# - Histograms
# - Location check
# - Anomaly detection
# - Missing data
# 
# 
# Data Preprocessing
# - Fill missing data
# - Rmove or replace anomaly data
# - Remove unusable data
# 
# Data Analysis
# *   Taxi fare system review by year:
#       - Current System:
#         - Initial charge: $2.50. (was $2)
#         - Mileage: 40 cents per 1/5 mile. (was 30 cents) - relate to estimated travel distance
#         - Waiting charge: 40 cents per 120 seconds. (was 30 cents per 90 seconds)
#         - JFK flat fare: $45. (was $35) - add if pick-up or drop-off point is inside JFK fare zone
#         - Newark surcharge: $15. (was $10) 4 p.m.–8 p.m. weekday - relate to date and time of data
#         - Surcharge: $1. (new charge) - add to all trips
#  
# *   Google Map API for travel distance estimation:
#     - By year?
#     - road infrastructure change over time?
# 
# 
# *   Correlation between fare and estimated travel distance

# In[1]:


import os
import pandas as pd
import datetime
import plotly
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import geopandas as gpd
import shapely
import osmnx as ox
import networkx as nx
import folium
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import statsmodels.api as sm


# # **01. Load Dataset from Local Drive**

# In[2]:


#C:\Users\ookum\Projects\API
#C:\Users\hayato.takasaki\AURECON\Tools\API
plotly_apikey_path = os.path.join(r'C:\Users\ookum\Projects\API', 'apikey_Plotly.txt')
plotly_apikey = open(plotly_apikey_path, 'r').read()
plotly.offline.init_notebook_mode(connected=True)


# In[3]:


#C:\Users\ookum\Projects\API
#C:\Users\hayato.takasaki\AURECON\Tools\API
mapbox_apikey_path = os.path.join(r'C:\Users\ookum\Projects\API', 'apikey_Mapbox.txt')
mapbox_apikey = open(mapbox_apikey_path, 'r').read()


# In[4]:


#C:\Users\ookum\Projects\Kaggle\01_NYC_TaxiFarePrediction\Dataset
#C:\Users\hayato.takasaki\AURECON\Kaggle\01_NYC_TaxiFarePrediction
in_path = r'C:\Users\ookum\Projects\Kaggle\01_NYC_TaxiFarePrediction\Dataset'
in_data_name = 'train.csv'


# In[5]:


file_list = os.listdir(in_path)


# In[6]:


for f in file_list:
    print(f)


# In[7]:


file_load = os.path.join(in_path, in_data_name)


# # 02. Convert Dataset into Dataframe

# ##### Steps:
# - Prepare functions for each process
# - Execute the functions within pandas chunk loop

# In[8]:


# Control Parameters
input_file = file_load
chunk_size = 100000


# In[9]:


# Step 1 - Load chunk of data
#for chunk in pd.read_csv(file_load, chunksize=chunk_size):
#    data processing here
#    output here
#return output


# In[19]:


df_train = pd.read_csv(file_load, nrows=10)
df_train


# In[20]:


# Step 2 - Preprocessing: Remove Column & Change Data Type
def dataframe_formatting(input_df, remove_col):
    temp_df = input_df.drop([remove_col], axis=1)
    temp_df[['fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']] = temp_df[['fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].astype(float)
    temp_df['passenger_count'] = temp_df['passenger_count'].astype(int)
    temp_df['pickup_datetime'] = pd.to_datetime(temp_df['pickup_datetime'])
    return temp_df


# In[21]:


df_train = dataframe_formatting(df_train, 'key')


# In[22]:


df_train.head()


# # 3. Data Preprocessing

# In[23]:


def process_datetime(input_df):
    temp_df = input_df.copy()
    temp_df['Year'] = temp_df['pickup_datetime'].map(lambda x: x.year)
    temp_df['Month'] = temp_df['pickup_datetime'].map(lambda x: x.month)
    temp_df['Day'] = temp_df['pickup_datetime'].map(lambda x: x.day)
    temp_df['Hour'] = temp_df['pickup_datetime'].map(lambda x: x.hour)
    temp_df.drop('pickup_datetime', axis=1, inplace=True)
    return temp_df


# In[24]:


def filter_by_coords(input_df, lon_min, lon_max, lat_min, lat_max):
    temp_df = input_df.copy()
    temp_df = temp_df[(temp_df.pickup_longitude > lon_min) & (temp_df.pickup_longitude < lon_max) & 
                  (temp_df.pickup_latitude > lat_min) & (temp_df.pickup_latitude < lat_max) &
                  (temp_df.dropoff_longitude > lon_min) & (temp_df.dropoff_longitude < lon_max) &
                  (temp_df.dropoff_latitude > lat_min) & (temp_df.dropoff_latitude < lat_max)]
    return temp_df


# In[25]:


def convert_to_absolute(input_df, col):
    input_df[col] = input_df[col].abs()
    return input_df


# In[26]:


df_train_pp = process_datetime(df_train)
df_train_pp = filter_by_coords(df_train_pp, -75, -72, 39, 41)
df_train_pp = convert_to_absolute(df_train_pp, 'fare_amount')
df_train_pp = df_train_pp.drop(['Year','Month','Day','Hour'], axis=1)


# In[27]:


len(df_train), len(df_train_pp)


# In[28]:


df_train_pp.head()


# In[29]:


df_train_pp.describe()


# # 5. Feature Engineering - Route Analysis 

# In[30]:


# Download OSM data
#NYC_network_graph = ox.graph_from_place('New York City', network_type='drive', simplify=False)
#ox.save_load.save_graphml(NYC_network_graph, filename='OSM_NewYorkCity_Streets.graphml', folder=net_path)
#NYC_network_graph.clear()


# In[31]:


net_file = 'OSM_NewYorkCity_Streets.graphml'
net_path = r'C:\Users\ookum\Projects\Kaggle\01_NYC_TaxiFarePrediction\Dataset\Additional_Dataset'
network_NYC = ox.save_load.load_graphml(net_file, folder=net_path)


# In[32]:


#ox.plot_graph(network_NYC, fig_height=10, fig_width=15, equal_aspect=True, bgcolor='grey', 
#              node_color='blue', node_size=5, edge_color='white', edge_linewidth=0.5);


# In[33]:


#def plot_shortest_route(in_graph, orig_lat, orig_lon, dest_lat, dest_lon):
#    orig_coord = (orig_lat, orig_lon)
#    dest_coord = (dest_lat, dest_lon)
#    orig_node = ox.get_nearest_node(in_graph, orig_coord)
#    dest_node = ox.get_nearest_node(in_graph, dest_coord)
#    taxi_route = nx.shortest_path(network_NYC, orig_node, dest_node, weight='length')
#    return taxi_route, ox.plot_route_folium(in_graph, taxi_route, route_color='red', route_width=2, route_opacity=0.8)


# In[34]:


## Test with single data
#pickup_lat = df_LReg['pickup_latitude'][501]
#pickup_lon = df_LReg['pickup_longitude'][501]
#dropoff_lat = df_LReg['dropoff_latitude'][501]
#dropoff_lon = df_LReg['dropoff_longitude'][501]
#route_result, route_plot = plot_shortest_route(network_NYC, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon);


# In[35]:


#route_plot


# In[36]:


def calculate_shortest_distance(in_graph, orig_lat, orig_lon, dest_lat, dest_lon):
    orig_coord = (orig_lat, orig_lon)
    dest_coord = (dest_lat, dest_lon)
    orig_node = ox.get_nearest_node(in_graph, orig_coord)
    dest_node = ox.get_nearest_node(in_graph, dest_coord)
    try:
        taxi_route_dist = nx.shortest_path_length(network_NYC, orig_node, dest_node, weight='length')
    except:
        taxi_route_dist = 0
    return taxi_route_dist


# In[37]:


def create_distance_feature(input_df, in_graph):
    if __name__ == '__main__':
        route_dist = joblib.Parallel(n_jobs=1)(joblib.delayed(calculate_shortest_distance)(in_graph, 
                                                                                           input_df.loc[i,'pickup_latitude'], 
                                                                                           input_df.loc[i,'pickup_longitude'], 
                                                                                           input_df.loc[i,'dropoff_latitude'], 
                                                                                           input_df.loc[i,'dropoff_longitude'] 
                                                                                           ) for i in input_df.index)
    
    return route_dist


# In[38]:


import time
start_t = time.clock()
list_dist = create_distance_feature(df_train_pp, network_NYC)
end_t = time.clock()
print(end_t-start_t)


# In[39]:


df_train_pp['Estimated_Distance'] = pd.Series(list_dist).values


# In[40]:


df_train_pp




# # 6. Linear Regression Model

# In[49]:


def remove_distance_0(input_df):
    temp_df = input_df.copy()
    try:
        temp_df = temp_df['Estimated_Distance'!=0].dropna()
    except:
        temp_df = temp_df
    return temp_df


# In[42]:


def minmax_scale_coords(input_df, from_col, to_col):
    temp_df = input_df.copy()
    mmsc = MinMaxScaler()
    temp_df.iloc[:,from_col:to_col] = mmsc.fit_transform(temp_df.iloc[:,from_col:to_col])
    return temp_df


# In[47]:


def feature_removal(input_df, features):
    temp_df = input_df.copy()
    temp_df.drop(features, axis=1, inplace=True)
    return temp_df


# In[50]:


df_train_pp = remove_distance_0(df_train_pp)
df_train_pp = minmax_scale_coords(df_train_pp, 1, 7)
df_train_fs = feature_removal(df_train_pp, ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'])


# In[51]:


df_train_fs.describe()


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(df_train_fs.iloc[:,1:], df_train_fs.iloc[:,0], test_size=0.2, train_size=0.8, random_state=42)
len(X_train),len(X_test), len(y_train), len(y_test)


# In[53]:


def multivariable_LinearRegression(X_training, y_training, X_testing, y_testing):
    #X_training = sm.add_constant(X_training)
    regresor_OLS = sm.OLS(endog=y_training, exog=X_training).fit()
    y_pred = regresor_OLS.predict(X_testing)
    result_df = pd.DataFrame([y_testing, y_pred]).T
    return regresor_OLS, regresor_OLS.summary(), result_df


# In[54]:


regresor_OLS, OLS_summary, result = multivariable_LinearRegression(X_train, y_train, X_test, y_test)
OLS_summary


# In[55]:


plt.scatter(x=result.iloc[:,0], y=result.iloc[:,-1]);


# # 7. Fare Prediction 

# In[56]:


# Load test dataset
test_data_folder = r'C:\Users\ookum\Projects\Kaggle\01_NYC_TaxiFarePrediction\Dataset'
test_data_file = 'test.csv'
df_test = pd.read_csv(os.path.join(test_data_folder, test_data_file))


# In[59]:


def testdataframe_formatting(input_df, remove_col):
    temp_df = input_df.drop([remove_col], axis=1)
    temp_df[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']] = temp_df[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].astype(float)
    temp_df['passenger_count'] = temp_df['passenger_count'].astype(int)
    temp_df['pickup_datetime'] = pd.to_datetime(temp_df['pickup_datetime'])
    return temp_df


# In[61]:


df_test = testdataframe_formatting(df_test, 'key')


# In[70]:


# Preprocessing
df_test_pp = process_datetime(df_test)
#df_test_pp = filter_by_coords(df_test_pp, -75, -72, 39, 41)
#df_test_pp = convert_to_absolute(df_test_pp, 'fare_amount')
df_test_pp = df_test_pp.drop(['Year','Month','Day','Hour'], axis=1)


# In[71]:


# Feature engineering
start_t = time.clock()
list_dist = create_distance_feature(df_test_pp, network_NYC)
end_t = time.clock()
print(end_t-start_t)


# In[73]:


#test_dist = pd.Series(list_dist)


# In[75]:


#test_dist.to_csv(os.path.join(r'C:\Users\ookum\Projects\Kaggle\01_NYC_TaxiFarePrediction\Dataset\Output','test_dist.csv'))


# In[77]:


df_test_pp['Estimated_Distance'] = pd.Series(list_dist).values


# In[80]:


# Feature scaling and selection
#df_test_pp = remove_distance_0(df_train_pp)
df_test_pp = minmax_scale_coords(df_test_pp, 1, 7)
df_test_fs = feature_removal(df_test_pp, ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'])


# In[81]:


y_pred_test = regresor_OLS.predict(df_test_fs)


# In[83]:


df_test = pd.read_csv(os.path.join(test_data_folder, test_data_file))
df_test.head()


# In[84]:


keys = df_test.iloc[:,0]


# In[100]:


df_submit = pd.concat([keys, y_pred_test], axis=1)
df_submit.columns = ['key','fare_amount']


# In[101]:


df_submit.head()


# In[102]:


df_submit.to_csv(os.path.join(r'C:\Users\ookum\Projects\Kaggle\01_NYC_TaxiFarePrediction\Dataset\Output','output_submit.csv'), index=False)


