"""
You are building an app for students in Kofu. There are 5 local cafes. 
We have a group of students who have rated these cafes from 1 to 5 stars. 
Some students haven't visited all the cafes yet (represented by 0).

Your goal is to build a system that finds a student most similar to you and recommends a cafe that you 
haven't been to yet. 
Find the user with the highest similarity to User 0.

Identify which cafe that user rated highly that User 0 has not visited (where User 0's rating is 0).

Use Boolean Masking to find these "unvisited" cafes.

"""
import numpy as np 
# User-Item dataset
# Each row is a user(User 0, User 1, ...) and each column is a cafe(Cafe 0, Cafe 1, ...)
# The values are the ratings given by each user, 0 means no rating
ratings = np.array([
    [5, 4, 0, 0, 2],  # You (User 0)
    [1, 2, 5, 4, 0],  # User 1
    [5, 3, 0, 1, 0],  # User 2
    [0, 0, 4, 5, 4]   # User 3
])

# Normalization to handle the users who haven't rated the cafes yet
# We subtract the mean rating of each user from their ratings 
ratings_normalized = ratings - ratings.mean(axis=1, keepdims=True)
print(f"Normalized ratings:\n{ratings_normalized}")

# Calcuate your normalized ratings
my_normalized_ratings = ratings_normalized[0]
print(f"Your normalized ratings: {my_normalized_ratings}")

# Calculate similarity scores
similarity_scores =  ratings_normalized @ my_normalized_ratings

print(f"Similarity scores: {similarity_scores}")

# User 2 has the highest similarity score with User 0 

# Identify which cafe that user rated highly that User 0 has not visited (where User 0's rating is 0).

# Find the cafes you haven't visited
unvisited_mask = ratings[0] == 0
print(f"Unvisited cafes: {unvisited_mask}")

# Apply your mask to User 2's ratings
user2_ratings = ratings[2]
recommended_ratings = user2_ratings[unvisited_mask]
print(f"Recommended ratings: {recommended_ratings}")


similarity_scores[0] = -np.inf # Used in machine learning to avoid self-recommendations, especailly in neural networks like "Attention Mechanisms"

# Find the index of highest remaining score
best_match_usr_idx = np.argmax(similarity_scores)

print(f"Best matching user: {best_match_usr_idx}")

# Extract the best recommendation 
# find ratings of most similar user i.e the best match 
best_match_user_ratings = ratings[best_match_usr_idx]
print(f"Best matching user ratings: {best_match_user_ratings}")
# Use np.where to create a list of candidates
# If unvisited keep rating, if visited  keep -1 so it's never the max
candidates = np.where(unvisited_mask, best_match_user_ratings, -1)
print(f"Candidates: {candidates}")

# Find index of highest rated unvisited cafe 
recommended_cafe_idx = np.argmax(candidates)
print(f"Recommended cafe index: {recommended_cafe_idx}")
cafe_list = ["Starbucks", "Tully's", "Local Roastery", "Library Cafe", "Bakery Cafe"]

print(f"Recommended cafe: {cafe_list[recommended_cafe_idx]}")

