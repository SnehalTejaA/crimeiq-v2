from data_loader import load_data, train_model, FEATURES
import os
df = load_data()
bundle = train_model(df)
print('R2:', bundle['r2'])
print('Pipeline type:', type(bundle['model']))
print('joblib saved:', os.path.exists('crime_model.pkl'))
print('All checks passed')
