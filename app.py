from flask import Flask, render_template, request
app = Flask(__name__)
import os
import requests
from titanic_demo import get_initial_page, get_data
from mlops.library import *
from datetime import datetime
index_path = "index.html"


@app.route("/")
def home():
    return render_template(index_path)

@app.route("/titanic_demo")
def demo_page():
    return get_initial_page()

@app.route('/titanic_demo/data', methods = ['POST'])
def data():
  startime = datetime.now()
  form_data = request.form
  page = get_data(form_data)
  endtime = datetime.now()
  print("loading debug - handling data took " + str(endtime - startime))
  return page


