from flask import Flask, jsonify, request
from sqlalchemy import create_engine
import pickle
import datetime
import pandas as pd

# Import and prepare variables
#uri = "postgresql://postgres:postgres@35.205.146.144/postgres"
uri="postgresql://user_pablo:HfUD2j0k2mSIj0FpEAoiJ5lxTPTNTgYg@dpg-d1j5lhili9vc739dp1q0-a.frankfurt-postgres.render.com/dbname_o7uj"
engine = create_engine(uri)
with open('my_model.pickle', 'rb') as file:
    model = pickle.load(file)

with open('my_scaler.pickle', 'rb') as file:
    scaler = pickle.load(file)

# Target names
target_names= ["setosa","versicolor","virginica"]

app = Flask(__name__)
app.config["DEBUG"] = True

# 1. Home page
@app.route('/', methods=['GET'])
def home():
    return "<h1>Iris predictions</h1><p>Making Iris predictions.</p>"

# 2. Make a prediction
@app.route('/api/v1/make_prediction_and_post', methods=['POST'])
def make_prediction():
    data = request.get_json()
    if (data['sepal length (cm)'], data['sepal width (cm)'],data['petal length (cm)'] ,data['petal width (cm)']):
        # Get features
        feature1 = float(data['sepal length (cm)'])
        feature2 = float(data['sepal width (cm)'])
        feature3 = float(data['petal length (cm)'])
        feature4 = float(data['petal width (cm)'])
        X = [[feature1, feature2, feature3, feature4]]
        X = scaler.transform(X)
        y_pred = model.predict(X)
        prediction = target_names[int(y_pred)]
        dict_X = [f"sepal length (cm): {data['sepal length (cm)']}",
                  f"sepal width (cm)': {data['sepal width (cm)']}",
                  f"petal length (cm)': {data['petal length (cm)']}",
                  f"petal width (cm)': {data['petal width (cm)']}"]
        my_json = {'datetime': str(datetime.datetime.now()),
                   'X': [dict_X],
                   'prediction': prediction}
        df = pd.DataFrame([my_json])
        df.to_sql('my_table', con=engine, if_exists='append', index=None)

        query = "SELECT * FROM my_table"
        df_answer = pd.read_sql(query, con=engine)

        return jsonify(df_answer.to_dict(orient='records'))
    else:
        return "No suitable params"
    
# Get all the database
@app.route('/api/v1/history', methods=['GET'])
def get_history():
    query = "SELECT * FROM my_table"
    df_answer = pd.read_sql(query, con=engine)

    return jsonify(df_answer.to_dict(orient='records'))



if __name__ == '__main__':
    # This ensures the Flask development server runs when you execute the script directly
    app.run()


