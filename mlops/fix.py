import sys
import library
#from library import CustomMappingTransformer, CustomTargetTransformer, CustomTukeyTransformer, CustomRobustTransformer, CustomKNNTransformer

#sys.modules['__main__'].CustomMappingTransformer = CustomMappingTransformer
#sys.modules['__main__'].CustomTargetTransformer = CustomTargetTransformer
#sys.modules['__main__'].CustomTukeyTransformer = CustomTukeyTransformer
#sys.modules['__main__'].CustomRobustTransformer = CustomRobustTransformer
#sys.modules['__main__'].CustomKNNTransformer = CustomKNNTransformer

import joblib

with open("final_fully_fitted_pipeline.pkl", "rb") as f:
    transformer = joblib.load(f)

# Now re-save correctly:
#joblib.dump(transformer, "clean_fitted_pipeline.pkl")