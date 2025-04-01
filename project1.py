#  HOUSE PRICE PREDICTION MODEL ----

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
from xgboost import XGBRegressor
import joblib

# Reading CSV File
data = pd.read_csv("AI ML LEARN\House Price Pred. Project\Housing.csv")
df = pd.DataFrame(data)

# Check Missing Values 
miss_ = df.isnull().sum()
if miss_.sum() == 0:
    print("No Missing Values In Dataset")

# Apply Label Encoding , Furnished-0 , SemiFurnished-1 , Unfurnished-2
encoder = LabelEncoder()
df['furnishingstatus'] = encoder.fit_transform(df['furnishingstatus'])

# For Yes-1 , No-0 , Handling Categorized Columns 
yes_no_handling = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in yes_no_handling:
    df[col] = df[col].map({'yes':1 , 'no' : 0}) # Map Apply Any Operation To All Iterable Objects

# Using Feature Engineering , Creating And Modifying Features Of Existing Dataset
df['price_per_sqft'] = df['price'] / df['area']
df['total_rooms'] = df['bedrooms'] + df['bathrooms'] + df['stories']
df['luxury_index'] = df['airconditioning'] + df['hotwaterheating'] + df['prefarea']

X = df.drop(columns=['price']) # Dropping Price From(Features) , Cause Its An Outcome Y
y = df['price'] # Outcome

# Feature Scalling 
scaler = StandardScaler()
X_ = scaler.fit_transform(X)


# Scatter Plot Diagram , Shows Relation Between Area And Price
plt.figure(figsize=(8 , 5))
sns.scatterplot(x=df['area'] , y=df['price'])
plt.title("Scatter Plot (Area , Price)")
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

# Heatmap Shows Colors (Red , Blue , White etc) , It Shows Numbers (Like 0.8, -0.3, 1.0, etc.) , 
# 1.0 Means "perfect relation", Means This Feature Affect The Price Most , Negative value Means "opposite relation"
plt.figure(figsize=(8,5))
sns.heatmap(df.corr() , annot=True , cmap='coolwarm' , fmt='.2f')
plt.title('Heatmap Of House Prediction Data')
plt.show()

# Splitting Data Into Train , Test
X_train , X_test , Y_train , Y_test = train_test_split(X_ , y , test_size=0.2 , random_state=42)

# Training Model 
model = XGBRegressor() # Best Accuracy in This Dataset Than RandomForest And Linear Regression
model.fit(X_train,Y_train)

# Saving Model After Train
joblib.dump(model, 'house_price_model.pkl')  
joblib.dump(scaler, 'scaler.pkl')  # Scaler Too
print("Model & Scaler Saved Successfully!")

y_pred = model.predict(X_test)
# Checking Model Accuracy 
print("MAE : " , mean_absolute_error(Y_test , y_pred))
print("MSE : " , mean_squared_error(Y_test , y_pred))
print("R2 SCORE : " , r2_score(Y_test , y_pred))

# Load Saved Model
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Take Input From User
def get_user_input():
    area = int(input("Enter The Area Of House: "))
    bed = int(input("How Many Bedrooms You Want: "))
    bath = int(input("How Many Bathrooms You Want: "))
    stories = int(input("How Many Stories You Want: "))
    mainroad = int(input("Main Road Access (1-Yes, 0-No): "))
    guestroom = int(input("Guest Room Available (1-Yes, 0-No): "))
    basement = int(input("Basement Available (1-Yes, 0-No): "))
    hotwater = int(input("Hot Water Heating (1-Yes, 0-No): "))
    ac = int(input("Number Of ACs: "))
    prefarea = int(input("Preferred Area (1-Yes, 0-No): "))
    furnishing = int(input("Furnishing Status (0-Furnished, 1-Semi, 2-Unfurnished): "))
    parking = int(input("Number of Parking Spaces: "))
    
    return [area, bed, bath, stories, mainroad, guestroom, basement, hotwater, ac, prefarea, furnishing, parking]


def predict_house_price():
    new_house =  get_user_input()  # get_user_input()
    if new_house[0] < 1650:  # Check if area is below dataset range
        print("Error: House area should be at least 1650 sqft for accurate prediction.")
        return
    
    # Compute Feature Engineering for New Input
    avg_price = df['price_per_sqft'].median()
    price_per_sqft = avg_price # For Assumption , Cause Its Price / Area , But The Price Will Get After Predicting
    total_rooms = new_house[1] + new_house[2] + new_house[3]
    luxury_index = new_house[8] + new_house[7] + new_house[9]
    
    # Convert to DataFrame with Correct Columns
    new_house.extend([price_per_sqft,total_rooms, luxury_index])
    new_house_df = pd.DataFrame([new_house], columns=['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 
        'basement', 'hotwaterheating', 'airconditioning','parking', 'prefarea', 
        'furnishingstatus' , 'price_per_sqft' ,'total_rooms', 'luxury_index'])
    
    # Scale Input
    new_house_scaled = scaler.transform(new_house_df)
    
    # Predict Price
    predicted_price = model.predict(new_house_scaled)
    print(f"Predicted House Price: {predicted_price[0]} /- Only ")
    # print(len(new_house))  # Check input column count
    # print(new_house)  # See what data is missing

predict_house_price()
# print(df[['area', 'price', 'price_per_sqft']].describe())