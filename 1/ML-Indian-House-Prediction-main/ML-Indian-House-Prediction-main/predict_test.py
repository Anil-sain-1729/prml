import requests


host = "127.0.0.1"
url = f'http://{host}:9696/predict'

data = {'State': 'telangana',
 'City': 'warangal',
 'Property_Type': 0.0,
 'BHK': 1,
 'Size_in_SqFt': 2059,
 'Price_per_SqFt': 0.24,
 'Year_Built': 1995,
 'Furnished_Status': 2.0,
 'Floor_No': 0,
 'Total_Floors': 26,
 'Age_of_Property': 30,
 'Nearby_Schools': 7,
 'Nearby_Hospitals': 6,
 'Public_Transport_Accessibility': 0.0,
 'Parking_Space': 'no',
 'Security': 0.0,
 'Amenities': 'garden, pool, gym, playground, clubhouse',
 'Facing': 2.0,
 'Owner_Type': 'broker',
 'Availability_Status': 'under_construction'}


response = requests.post(url, json=data).json()

print(f"Price (In Lakhs): {response['price']}")

