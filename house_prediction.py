import pandas as pd

df = pd.read_csv("bengaluru_house_prices.csv")  

print(df.head())
print(df.columns)
print(df.info())
df = df.drop("society", axis=1)
print(df.isnull().sum())
df = df.dropna()
print(df.shape)
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
print(df[['size', 'bhk']].head())
def convert_sqft(x):
    if '-' in x:
        tokens = x.split('-')
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None
df['total_sqft'] = df['total_sqft'].apply(convert_sqft)   
df = df.dropna() 
print(df['total_sqft'].head())
df['location'] = df['location'].apply(lambda x: x.strip())
location_count = df['location'].value_counts()
print(location_count)
location_less_than_10 = location_count[location_count <= 10]
df['location'] = df['location'].apply(
    lambda x: 'other' if x in location_less_than_10 else x
)
print(df['location'].value_counts())
df = df[df['total_sqft'] / df['bhk'] >= 300]
print(df.shape)
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
def remove_price_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = subdf['price_per_sqft'].mean()
        st = subdf['price_per_sqft'].std()
        reduced_df = subdf[(subdf['price_per_sqft'] > (m - st)) & 
                           (subdf['price_per_sqft'] < (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df = remove_price_outliers(df)
print(df.shape)
df = df.drop(['size', 'price_per_sqft','availability'], axis=1)
dummies = pd.get_dummies(df['location'])
df = pd.concat([df, dummies], axis=1)
df = df.drop('location', axis=1)
dummies_area = pd.get_dummies(df['area_type'])
df = pd.concat([df, dummies_area], axis=1)
df = df.drop('area_type', axis=1)
x = df.drop('price', axis=1)
y = df['price']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(x_train, y_train)

print(model.score(x_test, y_test))
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print(df['location'].unique())
