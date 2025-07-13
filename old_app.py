from flask import Flask, render_template, request
app = Flask(__name__)
import os
import requests
from mlops.library import *
from datetime import datetime
index_path = "index.html"

from mlops.helpers import fpage_template, create_template_page

"""
  return {
      "config": config,
      "lgb_table": lgb_table,
      "logreg_table": logreg_table,
      "knn_table": knn_table,
      "ann_table": ann_table,
      "pipline_docs_html": pipeline_docs_html,
      "fitted_transformer":fitted_transformer,
      "lgb_model": lgb_model,
      "logreg_model": logreg_model,
      "knn_model": knn_model,
      "ann_model": ann_model,
      "lime_explainer": lime_explainer
  }
"""
@app.route("/")
def home():
    return render_template(index_path)

@app.route("/titanic_demo")
def demo_page():
    fpage = create_template_page(my_data["config"], fpage_template, my_data["pipeline_docs_html"], my_data["lgb_table"], my_data["logreg_model"], my_data["knn_table"], my_data["ann_table"])
    return get_initial_page(fpage, my_data["config"], my_data["pipeline_docs_html"], my_data["lgb_table"], my_data["logreg_model"], my_data["knn_table"], my_data["ann_table"])

@app.route('/titanic_demo/data', methods = ['POST'])
def data():
  startime = datetime.now()
  form_data = request.form
  page = get_data(form_data, my_data["fpage"])
  endtime = datetime.now()
  print("loading debug - handling data took " + str(endtime - startime))
  return page


my_data = load_models()