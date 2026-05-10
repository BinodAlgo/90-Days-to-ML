# Messy Gym Data Challenge
import pandas as pd 
import numpy as np 
messy_gym_data = {
    'Member_ID': [101, 102, 103, 104, 105, 106],
    'Age': [22, 28, np.nan, 45, 31, np.nan],
    'Membership': ['Basic', 'Premium', 'Premium', 'Basic', 'Basic', 'Premium'],
    'Visits_Per_Month': [4, 15, 12, 2, 8, 20],
    'Total_Spent': [40, 150, 120, np.nan, 80, 200]
}

# Step 1. convert this dictionary into a DataFrame called 'gym_df' 
gym_df = pd.DataFrame(messy_gym_data)
print(f" Original Gym Data: \n{gym_df}\n")

# Step 2: Handle missing data
median_ages = gym_df['Age'].median()
gym_df['Age'] =gym_df['Age'].fillna(median_ages,inplace=True)
print(f"\nData after filling missing ages: \n{gym_df}\n") 


gym_df['Total_Spent'] = gym_df['Total_Spent'].fillna(40)
print(f"\nData after filling missing total spent: \n{gym_df}\n")


# Step 3: Gym wants to reward thier most dedicated members. Create a new 
# DataFrame called active_members containin people visiting gym more than 10 times
active_members = gym_df[gym_df['Visits_Per_Month'] > 10]
print(f"\nActive Members: \n{active_members}\n") 

# Step 4: Aggregation( Grouping): Gym owner wants to know if premium members 
# are actually spending more money on average. 
total_spent_by_membership = gym_df.groupby('Membership')['Total_Spent'].mean()
print(total_spent_by_membership)
 
