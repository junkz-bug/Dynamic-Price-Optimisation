import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle

df = pd.read_csv('retail_price.csv')
df = df.drop(columns=['total_price','product_id','month_year','product_name_lenght','product_description_lenght','product_photos_qty','product_weight_g','customers','weekday','weekend','holiday','month','year','s','volume'])

X = df.drop('unit_price',axis=1)
y = df['unit_price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=433)

ohe = OneHotEncoder()
ohe.fit(X[['product_category_name']])

column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['product_category_name']), remainder='passthrough')

model = RandomForestRegressor()

pipe = make_pipeline(column_trans,model)
pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)
score = r2_score(y_test,y_pred)
pickle.dump(pipe,open('Model.pkl','wb'))
print(df.dtypes)
