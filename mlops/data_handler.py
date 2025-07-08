#Helper function to change ann output into pairs of probabilities in Lime - use as is
import pandas as pd
import numpy as np
#I probably should pass all the models in but I'm using them as globals. My bad. Being lazy.
import titanic_demo
from mlops.library import *
from datetime import datetime


def get_prediction(row):

  

  assert len(row.shape)==2 and row.shape[0]==1, f'Expecting nested numpy array but got {row}'
  assert titanic_demo.logreg_model.n_features_in_ == len(row[0]), f'length mismatch with what was trained on and row to predict: {logreg_model.n_features_in_} and {len(row[0])}'

  #lgb
  lgb_raw = titanic_demo.lgb_model.predict_proba(row)  #predict last row, we just tacked on
  yhat_lgb = lgb_raw[:,1]

  #KNN
  knn_raw = titanic_demo.knn_model.predict_proba(row)
  yhat_knn = knn_raw[:,1]

  #logreg
  logreg_raw = titanic_demo.logreg_model.predict_proba(row)
  yhat_logreg = logreg_raw[:,1]


  #ANN
  yhat_ann = titanic_demo.ann_model.predict(row)[:,0]

  return [yhat_lgb, yhat_knn, yhat_logreg, yhat_ann]

def ann_proba(rows):
  yhat = titanic_demo.ann_model.predict(rows)
  result = [[1.0-p[0],p[0]] for p in yhat]  #wrangle into proba form
  x = np.array(result)
  return x

#Helper function to build dataframe for Lime results - use as is
def create_lime_table(the_explainer):
  the_probs = the_explainer.predict_proba.round(2)
  the_list = the_explainer.as_list()
  df = pd.DataFrame(columns=['Condition', 'Probs', "Contribution"])
  for i,row in enumerate(the_list):
    df.loc[i] = [row[0],the_probs,row[1]]
  return df

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
    startime = datetime.now()
    # Create DataFrame with columns in the expected order
    row_df = pd.DataFrame(columns=column_order)
    row_df.loc[0] = np.nan  # Add blank row
    # Process form values and fill the DataFrame
    for field_id, field_config in config.items():
        form_field = field_config["form_field"]
        column_name = field_config["column_name"]

        if form_field in columns and column_name in row_df.columns:
            # Apply the field's processing function and assign to the correct column
            processed_value = field_config["process"](columns[form_field])
            row_df.loc[0, column_name] = processed_value

    # Run pipeline
    row_transformed = fitted_transformer.transform(row_df)

    # Grab added row
    new_row = row_transformed.loc[0].to_list()
    new_row = np.array(new_row)
    new_row = np.reshape(new_row, (1,-1)) if len(new_row.shape)==1 else new_row

    # Get predictions
    yhat_lgb, yhat_knn, yhat_logreg, yhat_ann = get_prediction(new_row)
    end_time = datetime.now()
    print("loading debug - handling data took " + (end_time - startime))
    return new_row, yhat_lgb, yhat_knn, yhat_logreg, yhat_ann


def get_dataset_config():
    """
    Centralized configuration for Bank Loan dataset fields.
    This single function defines both form elements and data processing.

    Returns a dictionary where each key is a field name, and each value is
    a dictionary of properties for that field.
    """
    return {
        "age": {
            "form_field": "age_field",          # HTML name attribute
            "label": "Enter Age",               # Display label
            "type": "numeric",                  # Data type
            "placeholder": "Unkown",
            "input_type": "text",               # HTML input type
            "column_name": "person_age",               # DataFrame column name
            "process": lambda x: int(x) if x.isdigit() and int(x) > 0 else np.nan,
        },
        "gender": {
            "form_field": "gender_field",
            "label": "Choose a gender",
            "type": "categorical",
            "input_type": "select",
            "column_name": "person_gender",
            "process": lambda x: x if x != "Other" else np.nan,
            "options": {
                "male": "Male",
                "female": "Female",
                "other": "Other",
            }
        },
        "education": {
            "form_field": "education_field",
            "label": "Select highest education",
            "type": "categorical",
            "input_type": "select",
            "column_name": "person_education",
            "process": lambda x: x if x != "unknown" else np.nan,
            "options": {
                "unknown": "Unknown",
                'High School': 'High School',
                'Associate': 'Associate',
                'Bachelor': 'Bachelor',
                'Master': 'Master',
                'Doctorate': 'Doctorate'}

        },
        "income": {
            "form_field": "income_field",
            "label": "Enter Annual Income ($)",               # Display label
            "placeholder": "Unkown",
            "type": "numeric",                  # Data type
            "input_type": "text",               # HTML input type
            "column_name": "person_income",               # DataFrame column name
            "process": lambda x: int(x) if x.isdigit() and int(x) > 0 else np.nan,
        },
        "employment_experience": {
            "form_field": "employment_experience_field",
            "label": "Enter Employment Experience in Years",               # Display label
            "placeholder": "Unkown",
            "type": "numeric",                  # Data type
            "input_type": "text",               # HTML input type
            "column_name": "person_emp_exp",               # DataFrame column name
            "process": lambda x: int(x) if x.isdigit() and int(x) > 0 else np.nan,
        },
        "home_ownership": {
            "form_field": "home_ownerhip",
            "label": "Select Home Ownership",
            "type": "categorical",
            "input_type": "select",
            "column_name": "person_home_ownership",
            "process": lambda x: x if x != "unknown" else np.nan,
            "options": {
                "unknown": "Unknown",
                "MORTGAGE": "Mortgage",
                "RENT": "Rent",
                "OWN": "Own",
                "OTHER": "Other",
            }
        },
        "loan_amount": {
          "form_field": "loan_amount",
          "label": "Enter Loan Amount ($)",               # Display label
          "type": "numeric",         # Data type
          "placeholder": "Unkown",
          "input_type": "text",               # HTML input type
          "column_name": "loan_amnt",               # DataFrame column name
          "process": lambda x: int(x) if x.isdigit() and int(x) > 0 else np.nan,
      },
      "loan_intent": {
          "form_field": "loan_intent",
          "label": "Select Loan Intent",
          "type": "categorical",
          "input_type": "select",
          "column_name": "loan_intent",
          "process": lambda x: x if x != "unknown" else np.nan,
          "options": {
              "unknown": "Unknown",
              "PERSONAL":"Personal",
              "EDUCATION": "Education",
              "MEDICAL": "Medical",
              "VENTURE": "Venture",
              "HOME IMPROVEMENT": "Home Improvement",
              "DEBT CONSOLIDATION": "Debt Consolidation",
          }
        },
        "loan_interest_rate": {
          "form_field": "loan_interest_rate",
          "label": "Enter Loan Interest Rate (%)",               # Display label
          "type": "numeric",
          "placeholder": "Unkown",
          # Data type
          "input_type": "text",               # HTML input type
          "column_name": "loan_int_rate",               # DataFrame column name
          "process": lambda x: int(x) if x.isdigit() and int(x) > 0 else np.nan,
      },
        "loan_percent_income": {
          "form_field": "loan_percent_income",
          "label": "Enter Loan's Percentage of Income (Decimanl Equivalent Form)",               # Display label
          "placeholder": "1.0 > x > 0.0",
          "type": "numeric",                  # Data type
          "input_type": "text",               # HTML input type
          "column_name": "loan_percent_income",               # DataFrame column name
          "process": lambda x: int(x) if x.isdigit() and int(x) > 0 else np.nan,
      },
        "cb_person_cred_hist_length": {
          "form_field": "cb_person_cred_hist_length",
          "label": "Enter Credit Bureau's Record of Credit History Length in Years",               # Display label
          "type": "numeric",
          "placeholder": "Unkown",# Data type
          "input_type": "text",               # HTML input type
          "column_name": "cb_person_cred_hist_length",               # DataFrame column name
          "process": lambda x: int(x) if x.isdigit() and int(x) > 0 else np.nan,
      },
        "credit_score": {
          "form_field": "credit_score",
          "label": "Enter Credit Score",               # Display label
          "placeholder": "Unkown",
          "type": "numeric",                  # Data type
          "input_type": "text",               # HTML input type
          "column_name": "credit_score",               # DataFrame column name
          "process": lambda x: int(x) if x.isdigit() and int(x) > 0 else np.nan,
      },
        "previous_loan_defaults_on_file": {
            "form_field": "previous_loan_defaults_on_file",
            "label": "Previous Loan Defaults on File",
            "type": "categorical",
            "input_type": "select",
            "column_name": "previous_loan_defaults_on_file",
            "process": lambda x: x if x != "unknown" else np.nan,
            "options": {
                "unknown": "Unknown",
                "Yes": "Yes",
                "No": "No",
            }
    }
    }

