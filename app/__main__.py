import argparse

from flask import Flask
from .api import api_bp, add_resources
from .model import Model


# Flask configuration parameter
def configuration_parameter():
    parser = argparse.ArgumentParser(description="Boot parameters for flask.")
    parser.add_argument("--port", type=int, default=4060, help="Port number for the Flask server")
    parser.add_argument('--host', type=str, default='localhost', help="Host address for the Flask server")
    parser.add_argument('--enable-mschema','--enable_mschema', default=False, action='store_true',
                        help="Whether to enable the database connection feature and M-Schema")

    return parser.parse_args()


def mian():
    app = Flask(__name__)
    args = configuration_parameter()

    # args = parser.parse_args()

    # Load the model only once
    model = Model()

    # Register the API blueprint
    app.register_blueprint(api_bp)

    # Add the SQLQuery resource with the model instance
    add_resources(model, args)

    app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    mian()
