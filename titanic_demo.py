from joblib import load
from flask import Flask
from flask import request
import os
import tensorflow as tf
print("hi", flush=True)
import os
import requests
from joblib import load
import logging
import pandas as pd
from mlops.library import *
from mlops.helpers import *
from mlops.data_handler import *
import pickle
import sys
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.modules['__main__'].CustomMappingTransformer = CustomMappingTransformer
sys.modules['__main__'].CustomTargetTransformer = CustomTargetTransformer
sys.modules['__main__'].CustomTukeyTransformer = CustomTukeyTransformer
sys.modules['__main__'].CustomRobustTransformer = CustomRobustTransformer
sys.modules['__main__'].CustomKNNTransformer = CustomKNNTransformer
#sys.modules['__main__'].CustomANNTransformer = CustomANNTransformer
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
# Load model


lgb_model = load(lgb_model_path)
logreg_model = load(logreg_model_path)
knn_model = load(knn_model_path)
ann_model = tf.keras.models.load_model(ann_model_path, custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU()})
logreg_thresholds = pd.read_csv(f'{model_path}/logreg_thresholds.csv').round(2)
knn_thresholds = pd.read_csv(f'{model_path}/knn_thresholds.csv').round(2)
lgb_thresholds = pd.read_csv(f'{model_path}/lgb_thresholds.csv').round(2)
ann_thresholds = pd.read_csv(f'{model_path}/ann_thresholds.csv').round(2)

# Load fitted transformer
fitted_transformer = load(f"{model_path}/final_fully_fitted_pipeline.pkl")
joblib.dump(fitted_transformer, "final_fully_fitted_pipeline.pkl") # __main__.CustomMappingTransformer 
#must now become mlops.library.CustomMappingTransformer in the pickle file
with open(f"{model_path}/lime_explainer.pkl", 'rb') as file:
    lime_explainer = pickle.load(file)

with open(f"{model_path}/pipeline-documentation.md", 'r', encoding='utf-8') as file:
    pipe_md_content = file.read()


config = get_dataset_config()

pipeline_docs_html = get_pipeline_documentation(pipe_md_content)
lgb_table = lgb_thresholds.to_html(index=False, justify='center').replace('<td>','<td style="text-align: center;">')
logreg_table = logreg_thresholds.to_html(index=False, justify='center').replace('<td>','<td style="text-align: center;">')
knn_table = knn_thresholds.to_html(index=False, justify='center').replace('<td>', '<td style="text-align: center;">')
ann_table = ann_thresholds.to_html(index=False, justify='center').replace('<td>', '<td style="text-align: center;">')

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
fpage = create_template_page(config, fpage_template, pipeline_docs_html, lgb_table, logreg_table, knn_table, ann_table)

def get_initial_page():
  return create_page(fpage, lgb='', knn='', logreg='', ann='', ensemble='', row_data='',
                          lgb_lime_table='',
                           logreg_lime_table = '',
                           knn_lime_table = '',
                           ann_lime_table = '')


def get_data(form_data):
  #get predictions
  new_row, yhat_lgb, yhat_knn, yhat_logreg, yhat_ann = handle_data(form_data.to_dict(), fitted_transformer, config, feature_names)  #calling my own function here
  ensemble = (yhat_lgb[0]+yhat_knn[0]+yhat_logreg[0]+yhat_ann[0])/4.0
  lgb = np.round(yhat_lgb[0], 2)
  knn = np.round(yhat_knn[0], 2)
  logreg = np.round(yhat_logreg[0], 2)
  ann = np.round(yhat_ann[0], 2)
  ensemble = np.round(ensemble, 2)

  #handle lime stuff
  lgb_lime_table = ''
  logreg_lime_table = ''
  knn_lime_table = ''
  ann_lime_table = ''

  if lime_explainer:
    try:
      logreg_explanation = lime_explainer.explain_instance(new_row[0], logreg_model.predict_proba, num_features=len(feature_names))
      lime_df = create_lime_table(logreg_explanation)
      logreg_lime_table = lime_df.to_html(index=False, justify='center').replace('<td>','<td style="text-align: center;">')
    except Exception as e:
      logreg_lime_table = f"Error generating LogReg LIME: {e}"
      pass
    try:
      lgb_explanation = lime_explainer.explain_instance(new_row[0], lgb_model.predict_proba, num_features=len(feature_names))
      lime_df = create_lime_table(lgb_explanation)
      lgb_lime_table = lime_df.to_html(index=False, justify='center').replace('<td>', '<td style="text-align: center;">')
    except Exception as e:
      lgb_lime_table = f"Error generating LGB LIME: {e}"
      pass
    try:
      knn_explanation = lime_explainer.explain_instance(new_row[0], knn_model.predict_proba, num_features=len(feature_names))
      lime_df = create_lime_table(knn_explanation)
      knn_lime_table = lime_df.to_html(index=False, justify='center').replace('<td>', '<td style="text-align: center;">')
    except Exception as e:
      knn_lime_table = f"Error generating KNN LIME: {e}"
      pass
    try:
      ann_explanation = lime_explainer.explain_instance(new_row[0], ann_proba, num_features=len(feature_names))
      lime_df = create_lime_table(ann_explanation)
      ann_lime_table = lime_df.to_html(index=False, justify='center').replace('<td>', '<td style="text-align: center;">')
    except Exception as e:
      ann_lime_table = f"Error generating ANN LIME: {e}"
      pass

  #fill in fpage with results from models and Lime
  return create_page(fpage, lgb=lgb, knn=knn, logreg=logreg, ann=ann, ensemble=ensemble, row_data=str(form_data.to_dict()),
                           lgb_lime_table=lgb_lime_table,
                           logreg_lime_table = logreg_lime_table,
                           knn_lime_table = knn_lime_table,
                           ann_lime_table = ann_lime_table
                           )














