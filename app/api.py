from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from MSchema import schema_get

api_bp = Blueprint('api', __name__)
app = Api(api_bp)


class SQLQuery(Resource):
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def post(self):
        data = request.get_json()
        question = data.get('question')
        # Whether enable M-Schema
        print("========Enable M-Schema:", self.args.enable_mschema)
        db_schema = schema_get() if self.args.enable_mschema else data.get('db_schema')
        evidence = data.get('evidence', "")
        if not question or not db_schema:
            return jsonify({"error": "Question and database schema are required."})

        sql_query = self.model.generate_sql(question=question, db_schema=db_schema, evidence=evidence)
        return jsonify({"sql_query": sql_query})


# Add the SQLQuery resource to the Api instance
def add_resources(model, args):
    app.add_resource(SQLQuery, '/text2sql', resource_class_args=(model, args))
