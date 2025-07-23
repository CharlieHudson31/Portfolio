from flask import Flask, request, render_template, abort
import pandas as pd
import logging
import datetime
# Import models from the separate module
from mlops.demo import get_initial_page, get_data, load_models
from mlops_id3.id3 import loadAndTrain, runTest
import matplotlib.pyplot as plt
import os
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_models()
index_file = "index.html"
# Initialize Flask
DATASETS = {
    "dataset1": {
        "train": "mlops_id3/datasets/data_sets1/training_set.csv",
        "test": "mlops_id3/datasets/data_sets1/test_set.csv",
        "model": "mlops_id3/dataset1.model.txt",
        "plot_name": ""
    },
    "dataset2": {
        "train": "mlops_id3/datasets/data_sets2/training_set.csv",
        "test": "mlops_id3/datasets/data_sets2/test_set.csv",
        "model": "mlops_id3/dataset2.model.txt",
        "plot_name": ""
    },
    "mushroom dataset": {
        "train": "mlops_id3/datasets/mushrooms/agaricuslepiotatrain1.csv",
        "test": "mlops_id3/datasets/mushrooms/agaricuslepiotatest1.csv",
        "model": "mlops_id3/mushrooms.model.txt",
        "plot_name": "Mushrooms"
    }
}
app = Flask(__name__)
@app.route("/")
def home():
    return render_template(index_file)

@app.route("/bank_loan_demo")
def demo_page():
    return get_initial_page()

@app.route('/bank_loan_demo/data', methods = ['POST'])
def data():
  startime = datetime.datetime.now()
  form_data = request.form
  page = get_data(form_data)
  endtime = datetime.datetime.now()
  print("loading debug - handling data took " + str(endtime - startime))
  return page

@app.route("/id3_demo")
def id3():

    return render_template("id3.html", datasets=DATASETS.keys())


@app.route("/id3_demo/show_model/<dataset_name>")
def show_model(dataset_name):
    train = DATASETS[dataset_name]["train"]
    test = DATASETS[dataset_name]["test"]
    model = DATASETS[dataset_name]["model"]
    loadAndTrain(train, test, model)

    acc = runTest()
    if dataset_name not in DATASETS:
        abort(404)

    if not os.path.exists(model):
        return f"Model file not found for {dataset_name}: {model}", 404
    with open(model, "r") as f:
        model_text = f.read()
    return render_template("show_model.html", dataset_name=dataset_name, model_text=model_text, acc=acc)

@app.route('/id3_demo/plot_dataset/<dataset_name>')
def plot_dataset(dataset_name):
    # Save plot
    plot_path = f"static/{dataset_name}_plot.png"
    plot_name = DATASETS[dataset_name]["plot_name"]

    return render_template('plot_dataset.html', plot_path=plot_path, name=plot_name)
