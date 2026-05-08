import pandas as pd 
import numpy as np 
# Fitness dataset 
data = {
    'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
    'Protein_g': [150, 160, np.nan, 155, 170], # Missed Wednesday's log
    'Lift_Volume_kg': [5000, 5200, 0, 5100, 5400]
}
# Converting dictionary into a DataFrame
df = pd.DataFrame(data)
print(f"Fitness Data:\n{df}\n")

# Impute Fill the missing protein data with the median
median_protein = df['Protein_g'].median()
df['Protein_g'] = df['Protein_g'].fillna(median_protein)
print(f"Imputed Data:\n{df}\n") 


# The Math of Scale: Min-Max Normalization
# We compress every feature into a scale between 0.0 and 1.0.
# X_{scaled} = (X - X_min) / (X_max - X_min)
min_lift_volume = df['Lift_Volume_kg'].min()
max_lift_volume = df['Lift_Volume_kg'].max() 
df['Lift_Volume_normalized'] = (df['Lift_Volume_kg'] - min_lift_volume) / (max_lift_volume - min_lift_volume) 
print(f"Normalized Data:\n{df}")

# The Math of Text: One-Hot Encoding
# For categorical data like 'Workout Type'( Pull, Push, Legs), we cannot simply use the names.
# We create a binary column for each category.
df['Workout_Type'] = ['Push', 'Pull', 'Legs', 'Rest', 'Cardio']
# Convert text to binary columns
df_encoded = pd.get_dummies(df, columns=['Workout_Type'], drop_first=True) # pd.from_dummies() convert dummies to categorical dataframe
print(f"Encoded Data:\n{df_encoded}")