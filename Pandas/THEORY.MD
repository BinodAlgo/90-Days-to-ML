# Theory of Pandas and Statistics

Before any machine learning model can learn, we need to clean, filter and understand the data.

Pandas is a powerful library for data manipulation and analysis. It provides two main data structures:
1. **Series**: A one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, etc.).
2. **DataFrame**: A two-dimensional labeled data structure with columns of potentially different types. It can be thought of as a spreadsheet or a SQL table.

If we feed raw messy data into a model, it will not give the desired output. Garbage in garbage out (GIGO). 

## Feature Engineering & Data Preparation

Let's take a real world scenario of an app that tracks your muscle growth. We are building a model to predict future muscle growth based on diets and lifting habits. If the dataset is missing values, numbers on entierly different scales, and text data, the model will not be able to learn from the data. We'll need to fix all these messy stuff mathematically. 

### 1. The Math of Missing Data: Imputation 
If we forget to input the calorie or log data for a day, the model will not work with as the missing data are handled as `NaN` (Not a Number). To solve this, we use Central Tendency (Mean, Median, Mode) to guess the missing value. 
* **Scenario**: If you forgot to log calorie intake for a day, replacing it with 0 destroys the model, instead you can replace it with median because it won't be skewed by that one day you ate a massive cheat meal. 

### 2. The Math of Scale: Min-Max Normalization
Machine Learning algorithms (like Gradient Descent, which we will build later) use addition and multiplication. If your Calories feature is around 2500, and your Muscle_Gain feature is 0.5 kg, the algorithm will mathematically assume that Calories is 5000x more important simply because the number is bigger.

**The Math**: We compress every feature into a scale between 0.0 and 1.0.

$$
X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}
$$

### 3. The Math of Text: One-Hot Encoding
Machine Learning models cannot read text. They only read Tensors. If you have a category like "Workout Type" (e.g., Push, Pull, Legs), you must convert it to numbers.

**The Trap**: Do not convert them to 1, 2, 3. The math engine will assume 3 (Legs) is numerically "greater" than 1 (Push).

**The Math**: We convert categories into Orthogonal Vectors (matrices where only one element is 1 and the rest are 0). Let's say we have $N$ categories. We create an $N$-dimensional vector:
* Push = `[1, 0, 0]`
* Pull = `[0, 1, 0]`
* Legs = `[0, 0, 1]`