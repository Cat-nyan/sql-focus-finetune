from flask import Flask
from .api import api_bp, add_resources
from .model import Model


def create_app():
    app = Flask(__name__)

    # args = parser.parse_args()

    # Load the model only once
    model = Model()

    # Register the API blueprint
    app.register_blueprint(api_bp)

    # Add the SQLQuery resource with the model instance
    add_resources(model)

    return app
