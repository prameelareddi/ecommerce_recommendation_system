import pandas as pd
from Levenshtein import distance
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.neighbors import NearestNeighbors

def remove_duplicates(df):
  """
  Sorts the dataframe by title, eliminates duplicates using Levenshtein distance,
  and removes specific unwanted entries.
  """
  df = df.sort_values('title').reset_index(drop=True)
  df['lev'] = None
  for a in range(len(df)-1):
    if distance(df.iloc[a].title, df.iloc[a+1].title) <= 3:
      df.at[a, 'lev'] = distance(df.iloc[a].title, df.iloc[a+1].title)
  df = df[df['lev'].isnull()].reset_index(drop=True)
  df = df.drop([9572]).reset_index(drop=True)
  return df

def encode_text(df):
  """
  Loads a sentence transformer model, encodes plot synopses into numerical vectors,
  and reshapes the dataframe to have vectors as columns.
  """
  model = SentenceTransformer('all-MiniLM-L6-v2')
  df_ = df.copy()
  # Use a loop to avoid potential 'progress_apply' errors depending on environment
  for i in range(len(df_)):
    df_.loc[i, 'plot_synopsis'] = model.encode(df_.loc[i, 'plot_synopsis'])
  df_index = df_.pop('title')
  df_ = df_[['plot_synopsis']]
  df_ = pd.DataFrame(np.column_stack(list(zip(*df_.values))))
  df_.index = df_index
  return df_

def build_recommender(df_encoded):
  """
  Fits a nearest neighbors model on the encoded data.
  """
  nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(df_encoded)
  return nbrs

def find_closest_title(title, df_encoded):
  """
  Finds the closest title to a given input title using Levenshtein distance.
  """
  m = pd.DataFrame(df_encoded.index)
  m['lev'] = m['title'].apply(lambda x: distance(x, title))
  return m.sort_values('lev', ascending=True)['title'].iloc[0]

def recommend_movies(df_encoded, nbrs, title):
  """
  Finds and prints similar movies using the nearest neighbors model, handling
  cases where the exact title isn't found.
  """
  title = find_closest_title(title, df_encoded)
  distances, indices = nbrs.kneighbors([df_encoded.loc[title]])
  for index in indices[0][1:]:
    print(title, '->', df_encoded.index[index])

# Load data
df = pd.read_csv("mpst_full_data.csv")

# Remove duplicates
df = remove_duplicates(df)

# Encode text
df_encoded = encode_text(df)

# Build recommender
nbrs = build_recommender(df_encoded)

# Get recommendations for a movie
recommend_movies(df_encoded, nbrs, "Avengers")

