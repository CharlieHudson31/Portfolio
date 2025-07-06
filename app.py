from flask import Flask, request, render_template
import pandas as pd
import logging
import datetime
# Import models from the separate module
from mlops.demo import get_initial_page, get_data, load_models

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_models()
index_file = "index.html"
# Initialize Flask

app = Flask(__name__)
@app.route("/")
def home():
    return render_template(index_file)

@app.route("/titanic_demo")
def demo_page():
    return get_initial_page()

@app.route('/titanic_demo/data', methods = ['POST'])
def data():
  startime = datetime.datetime.now()
  form_data = request.form
  page = get_data(form_data)
  endtime = datetime.datetime.now()
  print("loading debug - handling data took " + str(endtime - startime))
  return page
