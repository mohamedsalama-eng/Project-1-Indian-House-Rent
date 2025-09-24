
# #### Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from thefuzz import process

df = pd.read_csv("Indian_House_Rent_Dataset.csv")

# #### Load the dataset

df.columns = df.columns.str.lower().str.replace(" ", "_")
df.head(10)
df.drop_duplicates()

df.info()

# Cleaning column "Posted On"

df['posted_on'] = pd.to_datetime(df['posted_on'])  #662
values = df['posted_on'].value_counts(normalize=True)              
size = df['posted_on'].isna().sum()                                                      
fill = np.random.choice(values.index , size= size , p = values.values)              
df.loc[df['posted_on'].isna() , 'posted_on']  = fill 

# Cleaning column "BHK"

df['bhk'].isna().sum()  # 783
df['bhk'] = df['bhk'].str.replace(" room" , "")
df['bhk'] = df['bhk'].str.replace(".0" , "")

values = df['bhk'].value_counts(normalize=True)              
size = df['bhk'].isna().sum()                                                      
fill = np.random.choice(values.index , size= size , p = values.values)              
df.loc[df['bhk'].isna() , 'bhk']  = fill                                          
df['bhk'] = df['bhk'].astype('int')

# Cleaning column "Size"

df['size'] = df['size'].str.replace("Square Feet" , "")
df['size'] = df['size'].str.replace(".0" , "")
df['size'] = df['size'].astype('float') 
q1 = df['size'].quantile(0.25)
q3 = df['size'].quantile(0.75)
IQR  = q3 - q1
upper = q3 + (IQR * 1.5)
lower = q1 - (IQR * 1.5)
df= df[(df['size'] > lower) & (df['size'] < upper)]

# Cleaning column "Floor"

# First extract numbers
pattern = r"(\d+)\s+out\s+of\s+\d+"
df['floor_number'] = df['floor'].str.extract(pattern)

# Then handle special cases
df.loc[df['floor'].str.contains('Ground', na=False), 'floor_number'] = '0'
df.loc[df['floor'].str.contains('Upper Basement', na=False), 'floor_number'] = '0.25'
df.loc[df['floor'].str.contains('Lower Basement', na=False), 'floor_number'] = '-0.75'
df['floor_number'] = df['floor_number'].fillna(-1)
df['floor_number'] = df['floor_number'].astype("float")

# Cleaning column "Area Locality"

df['area_locality'].isna().sum()   # 462
df['area_locality'] = df['area_locality'].str.lower()
df['area_locality'].unique()
df['area_locality'] = df['area_locality'].fillna('unknown place')

# Cleaning column "City"

df['city'] = df['city'].str.lower()
df['city'].unique()[100:1000]
cities = ["mumbai", "delhi", "bangalore", "kolkata", "chennai", "ahmedabad", "hyderabad", "pune", "surat", "kanpur", "jaipur", "lucknow", "nagpur", "indore", "patna", "vadodara", "ludhiana", "visakhapatnam", "pimpri-chinchwad", "thane", "varanasi", "srinagar", "aurangabad", "bhopal", "agra", "nashik", "meerut", "amritsar", "navi mumbai", "prayagraj", "howrah", "jabalpur", "gwalior", "coimbatore", "vijayawada", "jodhpur", "madurai", "raipur", "kota"]
for city_bad in df['city'].unique():
    if pd.notna(city_bad) and isinstance(city_bad, str):  # Check if it's a valid string
        match , score = process.extractOne(city_bad , cities)
        if score > 80 :
            df.loc[df['city'] == city_bad , 'city'] = match
df['city'].value_counts()

value = df['city'].value_counts(normalize=True)
size = df['city'].isna().sum()
fill = np.random.choice(value.index, size = size , p = value.values)
df.loc[df['city'].isna(), 'city' ] = fill
df['city'].value_counts()
df['city'] = df['city'].astype('category')

# Cleaning column "Furnishing Status"

value = df['furnishing_status'].value_counts(normalize=True)
size_nan = df['furnishing_status'].isna().sum()
fill = np.random.choice(value.index , size = size_nan , p = value.values)
df.loc[df['furnishing_status'].isna() , 'furnishing_status'] = fill

df['furnishing_status'] = df['furnishing_status'].astype('category')

# Cleaning column "Tenant Preferred"

category = ['bachelors/family', 'family', 'bachelors', 'couples']

for statue in df['tenant_preferred'].unique():
    if pd.notna(statue) and isinstance(statue , str):
        match , score = process.extractOne(statue , category)
        if score > 80:
            df.loc[df['tenant_preferred'] == statue , 'tenant_preferred'] = match

value = df['tenant_preferred'].value_counts(normalize=True)
size_nan = df['tenant_preferred'].isna().sum()
fill = np.random.choice(value.index , size = size_nan , p = value.values)
df.loc[df['tenant_preferred'].isna() , 'tenant_preferred'] = fill

df['tenant_preferred'] = df['tenant_preferred'].astype('category')

# Cleaning column "Bathroom"

df['bathroom'] = df['bathroom'].str.replace(' Bathrooms' , "")

value = df['bathroom'].value_counts(normalize=True)
size_nan = df['bathroom'].isna().sum()
fill = np.random.choice(value.index , size = size_nan , p = value.values)
df.loc[df['bathroom'].isna() , 'bathroom'] = fill

df['bathroom'] = df['bathroom'].astype('category')

# Cleaning column "Point of Contact"

value = df['point_of_contact'].value_counts(normalize=True)
size_nan = df['point_of_contact'].isna().sum()
fill = np.random.choice(value.index , size = size_nan , p = value.values)
df.loc[df['point_of_contact'].isna() , 'point_of_contact'] = fill

df['point_of_contact'] = df['point_of_contact'].astype('category')

df['area_type'] = df['area_type'].str.lower()
value = df['area_type'].value_counts(normalize=True)
size_nan = df['area_type'].isna().sum()
fill = np.random.choice(value.index , size = size_nan , p = value.values)
df.loc[df['area_type'].isna() , 'area_type'] = fill
df['area_type'] = df['area_type'].astype('category')

q1 = df['rent'].quantile(0.25)
q3 = df['rent'].quantile(0.75)
IQR  = q3 - q1
upper = q3 + (IQR * 1.5)
lower = q1 - (IQR * 1.5)
df= df[(df['rent'] > lower) & (df['rent'] < upper)]
sns.histplot(data = df , x = 'rent')
df.shape  

sns.heatmap(df.corr(numeric_only=True), annot=True)

sns.countplot(data= df , y = 'tenant_preferred')
df['city'].value_counts()
df[['city' , 'rent','size']].groupby('city').median()

df.head()


df['3_or_less_bhk'] = 0
df.loc[df['bhk'] < 3, '3_or_less_bhk'] = 1


q1 = df['size'].quantile(0.25)
q3 = df['size'].quantile(0.75)
max = df['size'].max()
min = df['size'].min()
labels = ['small' , 'medium' , 'large']
bins = [min , q1 , q3 , max]
df['size_type'] = pd.cut(df['size'] , labels=labels , bins = bins)
df


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df[['rent']])
df['normalize_rent'] = scaler.transform(df[['rent']])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df[['size']])
df['Standardization_size'] = scaler.transform(df[['size']])


df.drop(columns='floor')
df.to_csv("Cleaned_House_Rent_Dataset.csv")

