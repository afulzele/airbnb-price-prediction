#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# In[3]:


df = pd.read_csv("./airbnb-price-prediction.csv")


# In[4]:


pd.set_option('display.max_columns', None)
df = df.drop(['name', 'thumbnail_url', 'id', 'description', 'first_review', 'host_since', 'last_review', 'neighbourhood', 'host_response_rate'], axis=1)
df.head()


# # Assumptions
# Looking at the data I've got few assumptions:
# 
# 1 - Is the log_price competetive in the places(zipcodes) where there are lots of AirBnbs (We'll take apartments in LA)
# 
# 2 - Does reviews matter ? (We'll take the AirBnbs who have outstanding reviews(number and percentage))
# 
# 3 - If the AirBnb is more close to the center of the city, is it costly ? (We have the city name. We can google the center coordinates of the city. We'll calculate the distance using the formula given on https://kite.com/python/answers/how-to-find-the-distance-between-two-lat-long-coordinates-in-python)

# # Let's check the Dataset !!

# In[5]:


df.describe()


# This shows that there are maximum of 74111 records and some of the features have missing values (but these are just numerical features). 

# In[6]:


df.info()


# In[7]:


df.shape


# # Cleaning the data before analysis

# In[8]:


df.isnull().sum()


# In[9]:


fill_0 = lambda col: col.fillna(0)
num_df = df.select_dtypes(include=['float', 'int'])
num_df_lst = num_df.columns[num_df.isnull().sum() > 0]
df_clean_num = df[num_df_lst].apply(fill_0, axis=0)
df = pd.concat([df.drop(num_df_lst, axis=1), df_clean_num], axis=1)


# In[10]:


df.isnull().sum()


# In[11]:


df['zipcode'] = df['zipcode'].str.strip()
df['zipcode'].replace('', np.nan, inplace=True)


# In[12]:


cat_df = df.select_dtypes(include=['object'])
cat_df_lst = cat_df.columns[cat_df.isnull().sum() > 0]
df.dropna(subset=cat_df_lst, axis=0, inplace=True)


# In[13]:


df.isnull().sum()


# In[14]:


df.shape


# In[15]:


df = df[~df['zipcode'].str.contains('-', regex=False)]
df = df[~df['zipcode'].str.contains('\r', regex=False)]
df = df[~df['zipcode'].str.contains('Near', regex=False)]
df = df[~df['zipcode'].str.contains('1m', regex=False)]


# In[16]:


df['zipcode'] = df['zipcode'].astype(float)


# In[17]:


df.dtypes


# In[18]:


df['amenities'].head()


# # Cleaning the Amenities column

# In[19]:


def seperate_am(x):
    if '{' in x:
        x = x.replace('{', '')
    if '}' in x:
        x = x.replace('}', '')
    if '"' in x:
        x = x.replace('"', '')
    lst = x.split(',')
    
    return lst


# In[20]:


df_sep_am = df.copy()
df_sep_am['amenities'] = df_sep_am['amenities'].apply(seperate_am)
df = pd.concat([df.drop(['amenities'], axis=1), pd.get_dummies(df_sep_am['amenities'].apply(pd.Series), prefix='', prefix_sep='amenities_', drop_first=True).sum(level=0, axis=1)], axis=1)


# In[21]:


df.shape


# In[22]:


df.head()


# In[23]:


len(df.columns[df.columns.str.contains('amenities')])


# # Data Analysis

# We're going to selectively sample out data by taking in consideration that the Airbnbs in the dataset would be a perfect choice for a couple.

# In[32]:


df_choose = df[(df['accommodates']==2)&(df['bedrooms']==1)&(df['beds']==1)&(df['bathrooms']==1)
               &(df['property_type']=='Apartment')&(df['room_type']=='Entire home/apt')]


# In[33]:


df_choose.shape


# In[34]:


df_citymean = df_choose[['city', 'log_price']].groupby(['city'], as_index=False).mean().sort_values('log_price', ascending=False)
df_citymean


# The mean log_price in SF is higher than others

# In[35]:


ax = sns.barplot(data=df_citymean, x='log_price', y='city', ci=None)
ax.set(xlabel='Mean of Log Price', ylabel='City')
plt.show()


# In[36]:


df_citycount = df_choose[['city', 'log_price']].groupby(['city'], as_index=False).count().sort_values('log_price', ascending=False)
df_citycount.rename(columns = {'log_price':'count'}, inplace = True)
ax = df_choose['city'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
ax.set_ylabel('')


# And we have more data on NYC than on SF.

# In[37]:


max(df_choose['log_price'])


# In[38]:


min(df_choose['log_price'])


# In[39]:


plt.figure(figsize=(15,8))
property_types = df_choose['property_type'].value_counts()
((property_types/df_choose.shape[0])*100).plot(kind="bar")
plt.title("Percentage of property types")
plt.xlabel('Property Types')
plt.ylabel('Percentages')
plt.show()


# 67% of AirBnbs are Apartments

# In[40]:


df_count_each = df_choose.groupby(['property_type', 'room_type'])['room_type'].count().unstack('room_type')
df_count_each["sum"] = df_count_each.sum(axis=1)
df_count_each = df_count_each.sort_values('sum', ascending=False)
df_count_each.head()


# In[41]:


df_count_each.drop(['sum'],axis=1).plot(kind='bar', stacked=True, figsize=(15,8))


# From the above visualization we can clearly see the room_types in the starting four. Just to check out the whole data we'll stretch the count of each preperty_type to 100% and check out the room_types in all the property_types.

# In[42]:


df_test = df_choose.groupby(['property_type', 'room_type'])['room_type'].count().unstack('room_type')
df_test["sum"] = df_test.sum(axis=1)
df_test = df_test.sort_values('sum', ascending=False)
df_test = df_test.drop(['sum'],axis=1)
df_test = df_test.apply(lambda x: (x / df_test.sum(axis=1))*100)
df_test.plot(kind='bar', stacked=True, figsize=(15,8))
df_test.head()


# We can say that all the AirBnb Hostels have more of Shared rooms than private rooms. The visualization is clear.

# In[43]:


df[['instant_bookable', 'log_price']].groupby(['instant_bookable'], as_index=False).mean()


# In[44]:


df_choose[['number_of_reviews', 'log_price']].plot.scatter(y='number_of_reviews', x='log_price', legend=False, c='Green', figsize=(15,8))


# In[45]:


df_choose[['review_scores_rating', 'log_price']].groupby(['review_scores_rating'], as_index=False).mean().sort_values(by='review_scores_rating',ascending=False).plot.bar(x='review_scores_rating', y='log_price', figsize=(15,8))


# Coming back to prove our assumption

# In[47]:


df_choose['property_type'] = df_choose['property_type'].str.strip()

df_a_la = df_choose

more_airbnb = df_a_la.groupby(['zipcode']).zipcode.value_counts().nlargest(1)
more_airbnb_lst = list(more_airbnb.index.droplevel(level=0))

# less_airbnb = df_a_la.groupby(['zipcode']).zipcode.value_counts().nsmallest(3)
# less_airbnb_lst = list(less_airbnb.index.droplevel(level=0))

# df_more_airbnb = df_a_la[(df_a_la['zipcode'].isin(more_airbnb_lst))]
# df_less_airbnb = df_a_la[(df_a_la['zipcode'].isin(less_airbnb_lst))]

# df_compare_zipcodes = pd.concat([df_more_airbnb, df_less_airbnb], axis=0)


# In[48]:


df_more_airbnb.shape


# In[49]:


df_more_airbnb.head()


# In[50]:


df_more_airbnb[ 'log_price'].mean()


# In[51]:


max(df_more_airbnb['log_price'])


# In[52]:


min(df_more_airbnb['log_price'])


# In[53]:


df_more_airbnb[df_more_airbnb.columns[df_more_airbnb.columns.str.contains('amenities')]].head()


# In[54]:


all_amenities = df_more_airbnb.columns[df_more_airbnb.columns.str.contains('amenities')]
nec_amenities = ['amenities_Air conditioning', 'amenities_Cable TV', 'amenities_Carbon monoxide detector',
                 'amenities_Essentials', 'amenities_Free parking on premises', 'amenities_Internet', 'amenities_Kitchen',
                 'amenities_Pets allowed', 'amenities_Pool', 'amenities_TV', 'amenities_Wireless Internet', 'amenities_Hot water',
                 'amenities_Refrigerator']
other_amenities = list(set(all_amenities) - set(nec_amenities))

df_more_airbnb['amenities_other'] = df_more_airbnb[other_amenities].sum(axis=1)
df_more_airbnb = df_more_airbnb.drop(other_amenities, axis=1)


# In[55]:


df_more_airbnb['amenities_other'] = (df_more_airbnb['amenities_other'] > 0).astype(int)


# In[56]:


df_more_airbnb.head()


# In[57]:


plt.figure(figsize=(10,8))
sns.heatmap(df_more_airbnb.drop(['accommodates', 'zipcode', 'bathrooms', 'bedrooms', 'beds', 'amenities_other'],axis=1).corr(), vmin=-1);


# In[58]:


plt.figure(figsize=(10,8))
plt.bar(['max', 'min'], [max(df_more_airbnb['log_price']), min(df_more_airbnb['log_price'])], align='center')
plt.ylabel('log_price')


# # Conclusion - The log_prices are not competetive in the places based on zipcodes where there are a lot of Airbnbs.
# 
# The analysis above also answers my second question. No, the reviews doesn't matter. However the analysis was done on a little set of data, we'll still check this on larger dataset.

# # Assumption 3

# In[59]:


lst = ['NYC', 'SF', 'DC', 'LA']


# In[60]:


df_sel_city = df_choose[(df_choose['city']=='NYC')|(df_choose['city']=='SF')|(df_choose['city']=='DC')|(df_choose['city']=='LA')]


# In[61]:


R = 6373.0

centers = {
    'NYC':{
        'latitude': 40.71427,
        'longitute': -74.00597
    },
    'DC':{
        'latitude': 38.9072,
        'longitute': 77.0369
    },
    'LA':{
        'latitude': 34.0522,
        'longitute': 118.2437
    },
    'SF':{
        'latitude': 37.7749,
        'longitute': 122.4194
    }
}

def calculate_distance(city, x, y):    
    lat1 = math.radians(centers.get(city)['latitude'])
    lon1 = math.radians(centers.get(city)['longitute'])
    lat2 = math.radians(x)
    lon2 = math.radians(y)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

#     distance = np.sqrt((dlat)**2+(dlon)**2)

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return round(distance/1000,4)


# In[62]:


df_sel_city['center_dist_km'] = [calculate_distance(city, x, y) for city, x, y in zip(df_sel_city['city'], df_sel_city['latitude'], df_sel_city['longitude'])]


# In[63]:


df_sel_city.head()


# In[64]:


fig = plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(221)
df_sel_city[df_sel_city['city']=='NYC'].plot.scatter(y='log_price', x='center_dist_km', ax=ax1, legend=True, c='DarkBlue')
pop_a = mpatches.Patch(color='None', label='New York')
plt.legend(handles=[pop_a])

ax2 = fig.add_subplot(222)
df_sel_city[df_sel_city['city']=='SF'].plot.scatter(y='log_price', x='center_dist_km', ax=ax2, legend=True, c='Red')
pop_a = mpatches.Patch(color='None', label='SF')
plt.legend(handles=[pop_a])

ax3 = fig.add_subplot(223)
df_sel_city[df_sel_city['city']=='LA'].plot.scatter(y='log_price', x='center_dist_km', ax=ax3, legend=True, c='Green')
pop_a = mpatches.Patch(color='None', label='LA')
plt.legend(handles=[pop_a])

ax4 = fig.add_subplot(224)
df_sel_city[df_sel_city['city']=='DC'].plot.scatter(y='log_price', x='center_dist_km', ax=ax4, legend=True, c='Purple')
pop_a = mpatches.Patch(color='None', label='DC')
plt.legend(handles=[pop_a])


# In[65]:


df_sel_city[(df_sel_city.columns[~df_sel_city.columns.str.contains('amenities')])]


# # Predictive Model

# In[66]:


def get_other_dums(df, lst):
    for col in lst:
        try:
            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_')], axis=1)
        except:
            continue
    return df


# In[67]:


get_other_dums_lst = ['room_type', 'bed_type', 'cleaning_fee', 'city', 'host_has_profile_pic', 'host_identity_verified',
                     'instant_bookable', 'cancellation_policy', 'property_type']
df_all_dums = get_other_dums(df, get_other_dums_lst)


# In[68]:


df_all_dums.drop(df_all_dums[df_all_dums['zipcode']=='90036-2514'].index , inplace=True)
response_col = ['log_price']
X = df_all_dums.drop(response_col, axis=1)
y = df_all_dums[response_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ln_reg = LinearRegression()
ln_reg.fit(X_train, y_train) #Fit

#Predict using your model
y_test_preds = ln_reg.predict(X_test)
y_train_preds = ln_reg.predict(X_train)

#Score using your model
test_score = r2_score(y_test, y_test_preds)
train_score = r2_score(y_train, y_train_preds)


# In[69]:


print("The rsquared on the training data was {}.  The rsquared on the test data was {}.".format(train_score, test_score))


# In[ ]:




