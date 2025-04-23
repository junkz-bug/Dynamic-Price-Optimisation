import numpy as np
from flask import Flask, render_template,request
import pandas as pd
import pickle

model = pickle.load(open("Model.pkl",'rb'))
df = pd.read_csv('retail_price.csv')
app = Flask(__name__)

@app.route('/')
def index():
    product_names = sorted(df['product_category_name'].unique())
    return render_template('index.html',product_names=product_names)


@app.route('/predict',methods=['POST'])
def predict():
    product_name = request.form.get('product_name')
    quantity = request.form.get('qty')
    # total_price = request.form.get('total_price')
    freight_price= request.form.get('freight_price')
    product_score = request.form.get('product_score')
    comp_1 = request.form.get('comp_1')
    comp_2 = request.form.get('comp_2')
    comp_3 = request.form.get('comp_3')
    ps1 = request.form.get('ps1')
    ps2 = request.form.get('ps2')
    ps3 = request.form.get('ps3')
    fp1 = request.form.get('fp1')
    fp2 = request.form.get('fp2')
    fp3 = request.form.get('fp3')
    lag_price = request.form.get('lag_price')

    prediction = model.predict(pd.DataFrame([[product_name,quantity,freight_price,product_score,comp_1,ps1,fp1,comp_2,ps2,fp2,comp_3,ps3,fp3,lag_price]],
                                            columns=['product_category_name', 'qty', 'freight_price', 'product_score',
                                                    'comp_1', 'ps1', 'fp1', 'comp_2', 'ps2','fp2', 'comp_3', 'ps3', 'fp3', 'lag_price']))
    prediction = "{:.2f}".format(prediction[0])
    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True)