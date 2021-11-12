import set_weights

set_weights.get_files()

import torch
import detect_image
from flask import Flask, request

app = Flask(__name__)


@app.route("/", methods=["GET"])
def main():
    source = request.args["source"]
    with torch.no_grad():
        response = detect_image.detect(source=source)
    return response


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8081)
