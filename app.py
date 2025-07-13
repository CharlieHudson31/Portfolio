from flask import Flask, request, render_template
import pandas as pd
import logging

# Import models from the separate module
from mlops_v2 import test
from mlops_v2 import helpers
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

index_file = "index.html"
# Initialize Flask
app = Flask(__name__)
@app.route("/")
def home():
    return render_template(index_file)

@app.route("/titanic_demo")
def demo_page():
    return get_initial_page(fpage, my_data["config"], my_data["pipeline_docs_html"], my_data["lgb_table"], my_data["logreg_model"], my_data["knn_table"], my_data["ann_table"])

@app.route('/titanic_demo/data', methods = ['POST'])
def data():
  startime = datetime.now()
  form_data = request.form
  page = get_data(form_data, my_data["fpage"])
  endtime = datetime.now()
  print("loading debug - handling data took " + str(endtime - startime))
  return page
