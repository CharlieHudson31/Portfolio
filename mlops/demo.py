import joblib
#from flask import Flask
from flask import request
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU
import tensorflow as tf
print("hi", flush=True)
import requests
import logging
import pandas as pd
import sys
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import dill
from mlops.helpers import *
import numpy as np
from pathlib import Path
import traceback
import eli5
from eli5.sklearn import explain_prediction
import shap
BASE_DIR = Path(__file__).resolve().parent
feature_names = ['person_age', 'person_gender', 'person_education', 'person_income', 'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score','previous_loan_defaults_on_file']
model_path = "mlops"
lgb_model_name = 'lgb_model.joblib'
logreg_model_name = 'logreg_model.joblib'
knn_model_name = 'knn_model.joblib'
ann_model_name = 'ann_model.keras'
lgb_model_path = f"{model_path}/{lgb_model_name}"
logreg_model_path = f"{model_path}/{logreg_model_name}"
knn_model_path = f"{model_path}/{knn_model_name}"
ann_model_path = f"{model_path}/{ann_model_name}"

for file_path in [lgb_model_path, logreg_model_path, knn_model_path, ann_model_path]:
  this_path = Path(file_path)
  assert this_path.exists()
import mlops.mlops_library.library
# need to import these transformers directly to load final_fully_fitted_transformer.pkl
from mlops.mlops_library.library import CustomMappingTransformer, CustomTargetTransformer, CustomKNNTransformer, CustomTukeyTransformer, CustomRobustTransformer, dataset_setup
from flask import jsonify
#if __name__ == '__main__':
lgb_model = None
logreg_model = None
logreg_model = None
knn_model = None
ann_model = None
fitted_transformer = None
logreg_thresholds = None
knn_thresholds = None
ann_thresholds = None
model_explainer = None
pipe_md_content = None
lgb_thresholds = None
config = None
fpage = None
sample_data = None
def load_models():
  global lgb_model, logreg_model, knn_model, ann_model, fitted_transformer, model_explainer, pipe_md_content, ann_thresholds, knn_thresholds, lgb_thresholds, logreg_thresholds, config, fpage, sample_data
  if lgb_model and logreg_model and knn_model and ann_model and fitted_transformer and model_explainer and pipe_md_content and logreg_thresholds is not None:
        # Already loaded
        return

  # Load models
  if not lgb_model:
      lgb_model = joblib.load(lgb_model_path)
  if not logreg_model:
      logreg_model = joblib.load(logreg_model_path)
  if not knn_model:
      knn_model = joblib.load(knn_model_path)
  if not ann_model:
      print("past keras loading", flush=True)
      ann_model = tf.keras.models.load_model(ann_model_path, custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU()})

  # Load CSV thresholds
  if not logreg_thresholds:
      logreg_thresholds = pd.read_csv(f'{model_path}/logreg_thresholds.csv').round(2)
  if not knn_thresholds:
      knn_thresholds = pd.read_csv(f'{model_path}/knn_thresholds.csv').round(2)
  if not lgb_thresholds:
      lgb_thresholds = pd.read_csv(f'{model_path}/lgb_thresholds.csv').round(2)
  if not ann_thresholds:
      ann_thresholds = pd.read_csv(f'{model_path}/ann_thresholds.csv').round(2)

  print("done reading CSVs - loading debug", flush=True)

  # Load fitted transformer
  if not fitted_transformer:
      with open(f"{model_path}/final_fully_fitted_pipeline_new.pkl", 'rb') as file:
          fitted_transformer = joblib.load(file)

  # Load model explainer
  if not model_explainer:
      with open(f"{model_path}/model_explainer.pkl", 'rb') as file:
          model_explainer = dill.load(file)

  # Load pipeline documentation
  if not pipe_md_content:
      with open(f"{model_path}/pipeline-documentation.md", 'r', encoding='utf-8') as file:
          pipe_md_content = file.read()

  # Get pipeline config + HTML
  if not config:
      config = get_dataset_config()
      print("loading debug - got config", flush=True)

  pipeline_docs_html = get_pipeline_documentation(pipe_md_content)
  print("loading debug - got pipeline.md", flush=True)

  # Convert thresholds to HTML
  lgb_table = lgb_thresholds.to_html(index=False, justify='center').replace('<td>', '<td style="text-align: center;">')
  print("loading debug - got lgb table", flush=True)
  logreg_table = logreg_thresholds.to_html(index=False, justify='center').replace('<td>', '<td style="text-align: center;">')
  knn_table = knn_thresholds.to_html(index=False, justify='center').replace('<td>', '<td style="text-align: center;">')
  print("loading debug - one more table left", flush=True)
  ann_table = ann_thresholds.to_html(index=False, justify='center').replace('<td>', '<td style="text-align: center;">')
  print("loading debug - got all tables", flush=True)

  dataset_path = f"{model_path}/loan_data.csv"

  applicant_data_trimmed = pd.read_csv(dataset_path)
  len(applicant_data_trimmed)
  target = 'loan_status'
  N = 5000
  downsample_df = applicant_data_trimmed.groupby(target, group_keys=False).apply(lambda x: x.sample(int(np.rint(N*len(x)/len(applicant_data_trimmed))))).sample(frac=1).reset_index(drop=True)
  len(downsample_df)
  applicant_features = downsample_df.drop(columns=target)
  labels = downsample_df[target].to_list()
  target = 'loan_status'
  label_column = target #change to name of your label column

  with open(f"{model_path}/final_fully_fitted_pipeline_new.pkl", 'rb') as file:
      fitted_transformer = joblib.load(file)
  loan_applcation_rs = 176
  x_train,  x_test, y_train,  y_test = dataset_setup(downsample_df, label_column,fitted_transformer, rs=loan_applcation_rs)
  sample_data = shap.sample(x_train, 100)
 
  # Render front page
  if not fpage:
      page_template = get_fpage_template()
      fpage = create_template_page(
          config=config,
          fpage_template=page_template,
          pipeline_docs_html=pipeline_docs_html,
          lgb_table=lgb_table,
          logreg_table=logreg_table,
          knn_table=knn_table,
          ann_table=ann_table
      )
#fpage = helpers.create_template_page(my_data["config"], fpage_template, my_data["pipeline_docs_html"], my_data["lgb_table"], my_data["logreg_model"], my_data["knn_table"], my_data["ann_table"])
def get_prediction(row):
  assert len(row.shape)==2 and row.shape[0]==1, f'Expecting nested numpy array but got {row}'
  assert logreg_model.n_features_in_ == len(row[0]), f'length mismatch with what was trained on and row to predict: {logreg_model.n_features_in_} and {len(row[0])}'

  #lgb
  lgb_raw = lgb_model.predict_proba(row)  #predict last row, we just tacked on
  yhat_lgb = lgb_raw[:,1]

  #KNN
  knn_raw = knn_model.predict_proba(row)
  yhat_knn = knn_raw[:,1]

  #logreg
  logreg_raw = logreg_model.predict_proba(row)
  yhat_logreg = logreg_raw[:,1]

  #ANN
  yhat_ann = ann_model.predict(row)[:,0]

  return [yhat_lgb, yhat_knn, yhat_logreg, yhat_ann]


def handle_data(columns, fitted_transformer, config, column_order):
    """
    Process form data using the dataset configuration.

    Parameters:
    -----------
    columns : dict
        Dictionary containing form field values, with field names as keys
    fitted_transformer : Pipeline
        Fitted sklearn Pipeline for transforming the input data
    config : dict
        Dataset configuration from get_dataset_config()

    Returns:
    --------
    tuple
        (transformed_row, yhat_lgb, yhat_knn, yhat_logreg, yhat_ann)
    """

    startime = datetime.datetime.now()
    # Create DataFrame with columns in the expected order
    row_df = pd.DataFrame(columns=column_order)
    row_df.loc[0] = np.nan  # Add blank row
    print("Data types:", row_df.dtypes, flush=True)
    print("Raw input:", row_df.head(), flush=True)
    # Process form values and fill the DataFrame
    for field_id, field_config in config.items():
        form_field = field_config["form_field"]
        column_name = field_config["column_name"]

        if form_field in columns and column_name in row_df.columns:
            # Apply the field's processing function and assign to the correct column
            processed_value = field_config["process"](columns[form_field])
            row_df.loc[0, column_name] = processed_value

    # Run pipeline
    try:
      row_transformed = fitted_transformer.transform(row_df)
    except Exception as e:
          print("Tranforming row error:", e, flush=True)
          return jsonify({'error': 'Internal server error'}), 500
    # Grab added row
    new_row = row_transformed.loc[0].to_list()
    new_row = np.array(new_row)
    new_row = np.reshape(new_row, (1,-1)) if len(new_row.shape)==1 else new_row

    try:
    # Get predictions
      yhat_lgb, yhat_knn, yhat_logreg, yhat_ann = get_prediction(new_row)
      end_time = datetime.datetime.now()
      print("loading debug - handling data took " + str(end_time - startime), flush=True)
    except Exception as e:
        print("Prediction error:", e, flush=True)
        return jsonify({'error': 'Internal server error'}), 500
    return new_row, yhat_lgb, yhat_knn, yhat_logreg, yhat_ann

def get_initial_page():
  return create_page(fpage, lgb='', knn='', logreg='', ann='', ensemble='', row_data='',
                          lgb_explainer_table='',
                           logreg_explainer_table = '',
                           knn_explainer_table = '',
                           ann_explainer_table = '')


def get_data(form_data):
  #get predictions
  def ann_proba(rows):
    yhat = ann_model.predict(rows)
    result = [[1.0-p[0],p[0]] for p in yhat]  #wrangle into proba form
    x = np.array(result)
    return x
  new_row, yhat_lgb, yhat_knn, yhat_logreg, yhat_ann = handle_data(form_data.to_dict(), fitted_transformer, config, feature_names)  #calling my own function here
  ensemble = (yhat_lgb[0]+yhat_knn[0]+yhat_logreg[0]+yhat_ann[0])/4.0
  lgb = np.round(yhat_lgb[0], 2)
  knn = np.round(yhat_knn[0], 2)
  logreg = np.round(yhat_logreg[0], 2)
  ann = np.round(yhat_ann[0], 2)
  ensemble = np.round(ensemble, 2)
  

  lgb_explainer_table = ''
  logreg_explainer_table = ''
  knn_explainer_table = ''
  ann_explainer_table = ''

  if model_explainer:
    named_row_df = pd.DataFrame([new_row[0]], columns=feature_names)

    logreg_explanation = eli5.explain_prediction(logreg_model, named_row_df.iloc[0])
    logreg_explainer_table = eli5.format_as_html(logreg_explanation)

    lgb_explanation = eli5.explain_prediction(lgb_model, named_row_df.iloc[0])
    lgb_explainer_table = eli5.format_as_html(lgb_explanation)

    knn_explainer = shap.KernelExplainer(knn_model.predict_proba, sample_data)
    class_is_true_idx = 1
    knn_shap_values = knn_explainer.shap_values(new_row)
    knn_shap_values_single = knn_shap_values[0, :, class_is_true_idx]
    # Create a DataFrame
    knn_shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": new_row[0],
        "SHAP Value": knn_shap_values_single
    })
    # Convert to HTML
    knn_explainer_table = knn_shap_df.to_html(index=False)

    ann_explainer = shap.KernelExplainer(ann_proba, sample_data)
    ann_shap_values = ann_explainer.shap_values(new_row)
    ann_shap_values_single = ann_shap_values[0, :, class_is_true_idx]
    ann_shap_df = pd.DataFrame({
    "Feature": feature_names,
    "Value": new_row[0],
    "SHAP Value": ann_shap_values_single
    })

    # Convert to HTML
    ann_explainer_table = ann_shap_df.to_html(index=False)

  return create_page(fpage, lgb=lgb, knn=knn, logreg=logreg, ann=ann, ensemble=ensemble, row_data=str(form_data.to_dict()),
                           lgb_explainer_table=lgb_explainer_table,
                           logreg_explainer_table = logreg_explainer_table,
                           knn_explainer_table = knn_explainer_table,
                           ann_explainer_table = ann_explainer_table
                           )
