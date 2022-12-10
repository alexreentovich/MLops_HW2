from flask import Flask, request
from flask_restx import Resource, Api, fields, reqparse
from DictItem import DictItem
from RegressionObject import RegressionObject, app, api

regressionObject = RegressionObject()

my_description = "API for training ML models from sklearn.linear_model."

ns = api.namespace('Regressions', description=my_description)

resource_fields_train = api.model('Resource_train', {
    'Model_class': fields.String,
    'Hyperparam_dict': DictItem,
    'Data': DictItem
})

resource_fields_predict = api.model('Resource_predict', {
    'Data': DictItem
})

parser = reqparse.RequestParser()
parser.add_argument('id', type=int, help='Regression id', required=True)


@ns.route('/')
class RegressionList(Resource):
    def get(self):
        '''List all available model classes'''
        return 'Available model classes: any model class from \
sklearn.linear_model which supports fit and predict', 200


@ns.route('/MyRegression/Predict')
class MyRegressionPred(Resource):
    @ns.response(200, 'Prediction made')
    @ns.response(400, 'Something wrong with input json')
    @ns.response(404, 'Regression with given id does not exist')
    @ns.doc(body=resource_fields_predict, parser=parser)
    def post(self):
        '''Make prediction for given dataset and regression id.

           The payload must contain 'Data' - a dictionary which \
           can be properly read as pandas.DataFrame \
           ('Data' of required format is returned by pandas .to_dict()). \
            All columns in 'Data' must contain features. \
            'Data' must not contain NaN or non numeric values.
        '''
        args = parser.parse_args()
        id = args['id']
        return regressionObject.get_pred(id, api.payload)


@ns.route('/MyRegression/Train')
class MyRegressionTrain(Resource):
    @ns.response(200, 'Regression trained and saved')
    @ns.response(400, 'Something wrong with input json')
    @ns.doc(body=resource_fields_train)
    def put(self):
        '''Train regression on given dataset and save it.

           The payload must contain: 'Model_class' - a string \
            corresponding to some sklearn.linear_model; \
           'Hyperparam_dict' - a dictionary\
           which can be passed as kwargs; 'Data' - a dictionary which \
           can be properly read as pandas.DataFrame \
           ('Data' of required format is returned by pandas .to_dict()). \
            The first column in 'Data' is interpreted as \
            the target variable and the other columns are \
            treated as features. \
            'Data' must not contain NaN or non numeric values.
        '''
        return regressionObject.create(api.payload)

    @ns.response(200, 'Regression trained and saved')
    @ns.response(400, 'Something wrong with input json')
    @ns.doc(body=resource_fields_predict, parser=parser)
    def patch(self):
        '''Retrain regression stored under given id \
            on a new dataset and save it under old id.

        The payload must contain 'Data' - a dictionary which \
        can be properly read as pandas.DataFrame \
        ('Data' of required format is returned by pandas .to_dict()). \
        The first column in 'Data' is interpreted as \
        the target variable and the other columns are \
        treated as features. \
        'Data' must not contain NaN or non numeric values.
        '''
        args = parser.parse_args()
        id = args['id']
        return regressionObject.update(id, api.payload)


@ns.route('/MyRegression/Delete')
class MyRegressionDelete(Resource):
    @ns.response(204, 'Regression deleted')
    @ns.response(404, 'Regression with given id does not exist')
    @ns.doc(parser=parser)
    def delete(self):
        '''Delete a trained regression given its id'''
        args = parser.parse_args()
        id = request.args.get('id')
        return regressionObject.remove(id)
    

if __name__ == '__main__':
    app.run(debug=True)
