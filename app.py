from flask import Flask
app = Flask(__name__)
import os


port = int(os.environ.get("PORT", 5000))

@app.route("/")
def home():
    return "Hello from Railway!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
