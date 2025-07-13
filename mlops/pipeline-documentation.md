# Loan Application Data Pipeline Documentation

## Pipeline Overview
This pipeline preprocesses the loan application dataset to prepare it for machine learning modeling. It handles categorical encoding, outlier detection and treatment, and feature scaling.

<img src='https://github.com/CharlieHudson31/MLOPS/blob/main/loan_application_data_transformer_pipeline.png?raw=true' width='70%' alt='Pipeline Diagram'>

## Step-by-Step Design Choices

### 1. Gender Mapping (`person_gender`)
- **Transformer:** `CustomMappingTransformer('person_gender', {'male': 0, 'female': 1})),`
- **Design Choice:** Binary encoding of gender with female as 1 and male as 0
- **Rationale:** Simple categorical mapping that preserves the binary nature of the feature without increasing dimensionality.

### 2. Education Mapping (`person_education`)
- **Transformer:** `CustomMappingTransformer('person_education', {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}))`
- **Design Choice:** Ordinal encoding of education class from lowest (High School) to highest (Doctorate)
- **Rationale:** Preserves the ordering of education levels. This representation aligns with the assumption that higher education levels may positively influence loan approval likelyhood.

### 3. Target Encoding for Home Ownership Column (`person_home_ownership`)
- **Transformer:** `CustomTargetTransformer(col='person_home_ownership')`
- **Design Choice:** Target encoding.
- **Rationale:** 
  - Replaces the categorical 'person_home_ownership' feature with its relationship to the target variable

### 4. Target Encoding for Loan Intent Column (`loan_intent`)
  - **Transformer:** `CustomTargetTransformer(col='loan_intent')`
- **Design Choice:** Target encoding.
- **Rationale:** 
  - Replaces the categorical 'loan_intent' feature with its relationship to the target variable
  
### 5. Previous Loan Defaults on File Mapping (`previous_loan_defaults_on_file`)
- **Transformer:** ` CustomMappingTransformer('previous_loan_defaults_on_file', {'Yes': 1, 'No': 0}),`
- **Design Choice:** Binary encoding of whether the applicant has a previous loan default on file with yes as 1 and no as 0
- **Rationale:** Simple categorical mapping that preserves the binary nature of the feature without increasing dimensionality.

### 6. Outlier Treatment for Age (`person_age`)
- **Transformer:** `CustomTukeyTransformer(target_column='person_age', fence='outer'),`
- **Design Choice:** Tukey method with outer fence for identifying extreme outliers
- **Rationale:** Outer fence preserves most of the original distribution while handling extreme values

### 6. Outlier Treatment for Income (`person_income`)
- **Transformer:** `CustomTukeyTransformer(target_column='person_income', fence='outer'),`
- **Design Choice:** Tukey method with outer fence for identifying extreme outliers
- **Rationale:** Outer fence preserves most of the original distribution while handling extreme values

### 7. Outlier Treatment for Employment Experience (`person_emp_exp`)
- **Transformer:** `CustomTukeyTransformer(target_column='person_emp_exp', fence='outer')`
- **Design Choice:** Tukey method with outer fence for identifying extreme outliers
- **Rationale:** Outer fence preserves most of the original distribution while handling extreme values

### 8. Outlier Treatment for Loan Interest Rate (`loan_int_rate`)
- **Transformer:** `CustomTukeyTransformer(target_column='loan_int_rate', fence='outer')`
- **Design Choice:** Tukey method with outer fence for identifying extreme outliers
- **Rationale:** Outer fence preserves most of the original distribution while handling extreme values

### 9. Outlier Treatment for Loan Amount (`loan_amnt`)
- **Transformer:** `CustomTukeyTransformer(target_column='loan_amnt', fence='outer')`
- **Design Choice:** Tukey method with outer fence for identifying extreme outliers
- **Rationale:** Outer fence preserves most of the original distribution while handling extreme values

### 10. Outlier Treatment for Loan Percentage of Applicant Income (`loan_percent_income`)
- **Transformer:** `CustomTukeyTransformer(target_column='loan_percent_income', fence='outer')`
- **Design Choice:** Tukey method with outer fence for identifying extreme outliers
- **Rationale:** Outer fence preserves most of the original distribution while handling extreme values


### 11. Outlier Treatment for Credit History Length (`cb_person_cred_hist_length`)
- **Transformer:** `CustomTukeyTransformer(target_column='cb_person_cred_hist_length', fence='outer')`
- **Design Choice:** Tukey method with outer fence for identifying extreme outliers
- **Rationale:** Outer fence preserves most of the original distribution while handling extreme values

### 12. Outlier Treatment for Credit Score (`credit_score`)
- **Transformer:** `CustomTukeyTransformer(target_column='credit_score', fence='outer')`
- **Design Choice:** Tukey method with outer fence for identifying extreme outliers
- **Rationale:** Outer fence preserves most of the original distribution while handling extreme values

### 13. Age Scaling (`scale_age`)
- **Transformer:** `CustomRobustTransformer(target_column='person_age')`
- **Design Choice:** Robust scaling for Age feature
- **Rationale:**
  - Robust to outliers compared to standard scaling
  - Uses median and interquartile range instead of mean and standard deviation
  - Appropriate for Age which may not follow normal distribution

### 14. Income Scaling (`person_income`)
- **Transformer:** `CustomRobustTransformer(target_column='person_income')`
- **Design Choice:** Robust scaling for income feature
- **Rationale:**
  - Robust to outliers compared to standard scaling
  - Uses median and interquartile range instead of mean and standard deviation
  - Appropriate for income which typically follows a right-skewed distribution

### 15. Applicant Exployment Experience Scaling (`person_emp_exp`)
- **Transformer:** ` CustomRobustTransformer(target_column='person_emp_exp')`
- **Design Choice:** Robust scaling for employment experience feature
- **Rationale:**
  - Robust to outliers compared to standard scaling
  - Uses median and interquartile range instead of mean and standard deviation
  - Appropriate for employment experience which may not follow a normal distribution

### 16. Loan Interest Rate Scaling (`loan_int_rate`)
- **Transformer:** `CustomRobustTransformer(target_column='loan_int_rate')`
- **Design Choice:** Robust scaling for loan interest rate feature
- **Rationale:**
  - Robust to outliers compared to standard scaling
  - Uses median and interquartile range instead of mean and standard deviation
  - Appropriate for loan interest rate which may not follow a normal distribution


### 17. Loan Amount Scaling (`loan_amount`)
- **Transformer:** `CustomRobustTransformer(target_column='loan_amount')`
- **Design Choice:** Robust scaling for loan amount feature
- **Rationale:**
  - Robust to outliers compared to standard scaling
  - Uses median and interquartile range instead of mean and standard deviation
  - Appropriate for loan amount which may not follow a normal distribution

### 18. Loan Percentage of Applicant Income Scaling (`loan_percent_income`)
- **Transformer:** `CustomRobustTransformer(target_column='loan_percent_income')`
- **Design Choice:** Robust scaling for loan percentage of income feature
- **Rationale:**
  - Robust to outliers compared to standard scaling
  - Uses median and interquartile range instead of mean and standard deviation
  - Appropriate for loan percentage of income which may not follow a normal distribution

### 19. Credit History Length Scaling (`cb_person_cred_hist_length`)
- **Transformer:** `CustomRobustTransformer(target_column='cb_person_cred_hist_length')`
- **Design Choice:** Robust scaling for credit history length feature
- **Rationale:**
  - Robust to outliers compared to standard scaling
  - Uses median and interquartile range instead of mean and standard deviation
  - Appropriate for credit history length which may not follow a normal distribution


### 20. Credit History Length Scaling (`credit_score`)
- **Transformer:** `CustomRobustTransformer(target_column='credit_score')`
- **Design Choice:** Robust scaling for credit score feature
- **Rationale:**
  - Robust to outliers compared to standard scaling
  - Uses median and interquartile range instead of mean and standard deviation
  - Appropriate for credit score which may not follow a normal distribution
