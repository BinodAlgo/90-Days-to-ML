# In machine learning, we use panda to prepare and clean data
import pandas as pd 

#1. INTRODUCTION
#  Dataframe: A excel sheet, where rows and columns are present
# Series: A single column of data

data = { 
    'Name': ["Binod", "Laxmi", "Shikhar", "Prabin"],
    'Age': [25, 24, 23, 26],
    'City': ["Kathmandu", "Kolkata", "Mumbai", "Biratnagar"],
    'Height': [5.5, 5.4, 6.0, 5.8],
    'Salary':[40000, 90000, 50000, 55000]
}

df = pd.DataFrame(data) #Converting a python dictionary into a dataframe

print(f"Dataframe: \n{df}\n")

# Grabbing a single column: returns a series
names = df['Name']
print(f"Names: \n{names}\n") 

# 2. DATA SELECTION
# In numpy, we use numbers to index matrices/ arrays. In Pandas, we have columns names, so we use 
# column name, so we use loc(location by label) or iloc(integer location by number)
first_person = df.iloc[0] # Returns the first row
print(f"First Person: \n{first_person}\n")

# Slicing 
heights = df.iloc[1:3, 2] # We use integer slicing here
print(f"Heights: \n{heights}\n")

# Another way using loc 
second_person = df.loc[1]
print(f"Second Person: \n{second_person}\n")

# Slicing using loc
cities = df.loc[:, 'City']
print(f"Cities: \n{cities}\n")


# 3. BOOLEAN FILTERING
#  Just like in Numpy, we use boolean mask to filter DataFrames 
# without writing for loops. 
lower_salaries_mask = df['Salary'] <= 50000
# Apply the mask to DataFrame
poor_people_df = df[lower_salaries_mask]
print(f"\nPoor People Data (Salary <= 50000): \n{poor_people_df}\n")



# 4. DATA AGGREGATION AND GROUPING
# Let's add a Department Column 
df['Department'] = ['HR', 'Engineering', 'Marketing', 'Sales']
# What is the average salary per department?
avg_salary_per_dept = df.groupby('Department')['Salary'].mean()
df['Average Salary'] = df['Department'].map(avg_salary_per_dept)
print(f"\n{df}")

# 5. MERGING DATASETS 
# Let's create another dataframe
df_bonus = pd.DataFrame({
    'Name': ["Binod", "Laxmi"],
    'Bonus': [5000, 10000]
})
print(f"\nBonus Data: \n{df_bonus}")

# Merge df with df_bonus
df_merged = pd.merge(df, df_bonus, on='Name', how='left')
print(f"\nMerged Data: \n{df_merged}")

# 6. PIVOT TABLES
# Pivot tables are used to summarize data 
pivot_table = df.pivot_table(index='Department', values='Salary', aggfunc='mean')
print(f"\nPivot Table: \n{pivot_table}")