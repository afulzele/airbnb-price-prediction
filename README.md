# Airbnb price prediction
Airbnb is an online marketplace that connects people who want to rent out their homes with people who are looking for accommodations in that locale.

### Table of content
1. Assumptions for Analysis
2. Exploring the Dataset
3. Cleaning the Data
4. Data Analysis
5. Prediction

### Files
1. airbnb-price-prediction.ipynb is a Jupyter Notebook which contains the code for analysis and prediction.
2. airbnb-price-prediction.py is a Python the python version of the above file.
3. The dataset has been downloaded from Kaggle. [(link)](https://www.kaggle.com/stevezhenghp/deloitte-airbnb-price-prediction)

### Analysis
To explore the dataset we need to clean the data first. Before cleaning the data we need to drop few attributes `name, thumbnail_url, id, description, first_review, host_since, last_review, neighbourhood, host_response_rate`. We'll select all the numerical attributes and fill them with `0` since these missing values can not be filled with average. We could drop the rows too. We'll then remove unwanted zipcode and and convert the zipcode to int.

The `amenities` column needs to be cleaned as well. We'll seperate the string and make columns for each. We'll select few of the important amenities and sum up the rest of the amenities as `amenities_other`. The `amenities_other` is then converted into boolean although the sum before coverting to boolen would always be more than 0 and hence the boolean would then be 1 always. We can drop this column as well.

The obvious observation is that the log_price would always increase as the number of accommodates/bedrooms/beds/bathrooms increases. So we'll selectively choose the above attributes so that the airbnbs dataset becomes a reasonable choice for a couple ie by selecting data which has `accommodates=2, bedrooms=1,beds=1 and bathrooms=1`

### Assumptions and process
1. **If a Airbnb has more ratings/number of reviews then it would cost more?**

For the first assumption we'll create a scatter plot based on `number_of_reviews` and `log_price` as well as a bar plot with x axis as `rating` and y axis as `mean log_price of each rating`.

2. **Can we find cheap Airbnbs if we move away from the city center?**

We'll use the haversine distance formula to calculate the distance between each Airbnb and save it in a new column, `center_dist_km`. We'll use scatter plot to check the relation.

3. **Is there competition between Airbnbs in same area?**

We'll find the largest count of the Airbnb in NYC and check the maximum `log_price` and minimum `log_price`. If the price is in the same range then the Airbnbs are in competition.
