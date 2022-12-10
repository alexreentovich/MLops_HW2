from flask import Flask
from flask_restx import Api
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
from sklearn import linear_model
import os
import pickle
from dotenv import load_dotenv

d = os.getcwd()
load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = "mybestsecret"
app.config["SQLALCHEMY_DATABASE_URI"] = f"postgresql://postgres:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_DB')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
api = Api(app)

class ML_model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model = db.Column(db.LargeBinary, unique=True, nullable=False)

db.create_all()

class RegressionObject(object):
    def get_pred(self, id, request):
        '''Make prediction for given dataset contained in \
            request and regression id'''
        my_query = ML_model.query.get(id)
        if my_query is not None:
            parsed = request
            if 'Data' not in set(parsed.keys()):
                api.abort(400, "Wrong input format")
            X_test = pd.DataFrame(parsed['Data'])
            regr = pickle.loads(my_query.model)
            if X_test.shape[1] != len(regr.coef_):
                api.abort(400, "Data contains wrong number of features")
            if X_test.isnull().values.any():
                api.abort(400, "Data contains NaN")
            if not X_test.applymap(np.isreal).values.all():
                api.abort(400, "Data contains non numeric values")
            try:
                pred = regr.predict(X_test)
            except Exception as e:
                api.abort(400, f"Sklearn raised Exception '{e.args[0]}'. \
                          Try another input")
            return pd.DataFrame(pred).to_json(), 200
        else:
            api.abort(404, "Regression {} doesn't exist".format(id))

    def create(self, request):
        '''Train regression on given dataset with given hyperparameters \
            (contained in request) and save it'''
        parsed = request
        required_keys = set(['Data', 'Model_class', 'Hyperparam_dict'])
        if set(parsed.keys()) != required_keys:
            api.abort(400, "Wrong input format")
        input_dataframe = pd.DataFrame(parsed['Data'])
        if input_dataframe.isnull().values.any():
            api.abort(400, "Data contains NaN")
        if not input_dataframe.applymap(np.isreal).values.all():
            api.abort(400, "Data contains non numeric values")
        X_train = input_dataframe.iloc[:, 1:]
        y_train = input_dataframe.iloc[:, 0]
        try:
            regr = getattr(linear_model, parsed['Model_class'])
            regr = regr(**parsed['Hyperparam_dict'])
            regr.fit(X_train, y_train)
        except Exception as e:
            api.abort(400, f"Sklearn raised Exception '{e.args[0]}'. \
                      Try another input")
        new_model = ML_model(model=pickle.dumps(regr))
        db.session.add(new_model)
        db.session.commit()
        return f'Regression successfully trained and saved under id {new_model.id}', 200

    def update(self, id, request):
        '''Retrain regression with given id on a \
            new dataset contained in request'''
        parsed = request
        required_keys = set(['Data'])
        if set(parsed.keys()) != required_keys:
            api.abort(400, "Wrong input format")
        input_dataframe = pd.DataFrame(parsed['Data'])
        if input_dataframe.isnull().values.any():
            api.abort(400, "Data contains NaN")
        if not input_dataframe.applymap(np.isreal).values.all():
            api.abort(400, "Data contains non numeric values")
        X_train = input_dataframe.iloc[:, 1:]
        y_train = input_dataframe.iloc[:, 0]
        my_query = ML_model.query.get(id)
        if my_query is not None:
            regr = pickle.loads(my_query.model)
            try:
                regr = regr.fit(X_train, y_train)
            except Exception as e:
                api.abort(400, f"Sklearn raised Exception '{e.args[0]}'. \
                      Try another input")
            my_query.model = pickle.dumps(regr)
            db.session.commit()
            return f'Regression {id} successfully retrained', 200
        else:
            api.abort(404, "Regression {} doesn't exist".format(id))

    def remove(self, id):
        '''Delete a trained regression given its id'''
        my_query = ML_model.query.get(id)
        if my_query is not None:
            ML_model.query.filter_by(id=id).delete()
            db.session.commit()
            return f'Regression {id} successfully deleted', 204
        else:
            api.abort(404, "Regression {} doesn't exist".format(id))
