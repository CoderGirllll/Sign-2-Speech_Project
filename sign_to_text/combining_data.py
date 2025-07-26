import pandas as pd
import os

csv_files = ['Yes_data.csv', 'No_data.csv', 'Hello_data.csv', 'Thank You_data.csv', 'Please_data.csv', 'I Love You_data.csv']
combined_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

combined_df.to_csv('final_training_data.csv', index=False)
