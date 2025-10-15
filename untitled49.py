

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
movies = pd.read_csv('/content/tmdb_5000_movies.csv')
credits = pd.read_csv('/content/tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
print("Merged DataFrame Shape:", movies.shape)

import ast  # To safely evaluate literal expressions from string representation

# Select relevant features
features = ['genres', 'keywords', 'cast', 'crew']

# Drop rows with any missing values in the selected features
movies.dropna(subset=features, inplace=True)

# Helper function to convert JSON string to a list of names
def convert(obj):
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except (ValueError, SyntaxError) as e:
        print(f"Error processing: {obj}, Error: {e}")
        # Return an empty list or handle as appropriate
        return []
    return L

# Apply the conversion to genres and keywords
for feature in ['genres', 'keywords']:
    movies[feature] = movies[feature].apply(convert)

# For 'cast', we'll only take the first 3 actors
def get_top_cast(obj):
    L = []
    counter = 0
    try:
        for i in ast.literal_eval(obj):
            if counter < 3:
                L.append(i['name'])
                counter += 1
            else:
                break
    except (ValueError, SyntaxError) as e:
        print(f"Error processing: {obj}, Error: {e}")
        # Return an empty list or handle as appropriate
        return []
    return L

movies['cast'] = movies['cast'].apply(get_top_cast)

# For 'crew', we'll only extract the director's name
def get_director(obj):
    L = []
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except (ValueError, SyntaxError) as e:
        print(f"Error processing: {obj}, Error: {e}")
        # Return an empty list or handle as appropriate
        return []
    return L

movies['crew'] = movies['crew'].apply(get_director)

# Remove rows where the error occurred (crew or cast could not be processed)
movies = movies[movies['crew'].apply(lambda x: isinstance(x, list) and len(x) > 0)].reset_index(drop=True)
movies = movies[movies['cast'].apply(lambda x: isinstance(x, list) and len(x) > 0)].reset_index(drop=True)


print("\nSample of Processed Features:")
print(movies[['title', 'genres', 'cast', 'crew', 'keywords']].head(2))

# Helper function to join list elements with a unique separator
def process_list(L):
    return " ".join(L)

# Apply the function to all list-based features and create the 'soup'
# Filter out rows where all feature lists are empty before creating 'soup'
valid_movies = movies[movies['keywords'].apply(lambda x: len(x) > 0) |
                      movies['cast'].apply(lambda x: len(x) > 0) |
                      movies['genres'].apply(lambda x: len(x) > 0) |
                      movies['crew'].apply(lambda x: len(x) > 0)].copy() # Use .copy() to avoid SettingWithCopyWarning

valid_movies['soup'] = valid_movies['keywords'].apply(process_list) + ' ' + \
                       valid_movies['cast'].apply(process_list) + ' ' + \
                       valid_movies['genres'].apply(process_list) + ' ' + \
                       valid_movies['crew'].apply(process_list)

valid_movies['soup'] = valid_movies['soup'].str.lower()

# Print the first few entries of the 'soup' column of valid_movies for inspection
print("\nSample of 'soup' column in valid_movies:")
print(valid_movies['soup'].head())


# Use TF-IDF to create a matrix of features (temporarily without stop words for debugging)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(valid_movies['soup'])

print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)

# Update the 'movies' DataFrame to the filtered version for subsequent steps
movies = valid_movies

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a Series that maps a movie title to its index
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim, movies=movies, indices=indices):
    if title not in indices:
        return f"Sorry, the movie '{title}' is not in our database."

    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the titles of the top 10 most similar movies
    return movies['title'].iloc[movie_indices]

# Test the recommender with a movie title
print("\nRecommendations for 'The Dark Knight Rises':")
recommendations = get_recommendations('The Dark Knight Rises')
print(recommendations)