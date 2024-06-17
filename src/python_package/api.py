from flask import Flask, jsonify, request
from flask_restx import Api, Resource
from werkzeug.exceptions import HTTPException

from python_package.__init__ import __version__
from python_package.main import hello_world

# from python_package.utils.api_doc import set_swagger_fields


app = Flask(__name__)

api = Api(
    app,
    version=__version__,
    title="TESEO Apiprocess",
    description="Application Program Interface to perform online TESEO simulations.",
    default="TESEO Apiprocess",
    default_label="main methods",
)

# api, fields_input_json = set_swagger_fields(api)



@api.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    return {
        "code": e.code,
        "name": e.name,
        "description": e.description,
    }, e.code


@api.route("/check-api")
class check_api(Resource):
    def get(self):
        return jsonify("This Flask-Api is up!")


@api.route("/say-hello", methods=["POST"])
class say_hello(Resource):
    def post(self):
        return jsonify(hello_world(request.json["word"]))

