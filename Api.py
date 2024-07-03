from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load the dataset
data = pd.read_csv('data.csv')

# Remove unnecessary column
data = data.drop('id', axis=1)

# Define tokenizer and stemmer
stemmer = SnowballStemmer('english')


def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stems = [stemmer.stem(t) for t in tokens]
    return stems


# Create stemmed tokens column
data['stemmed_tokens'] = data.apply(lambda row: tokenize_and_stem(row['Title'] + ' ' + row['Description']), axis=1)

# Define TF-IDF vectorizer and cosine similarity function
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)


def cosine_sim(text1, text2):
    text1_concatenated = ' '.join(text1)
    text2_concatenated = ' '.join(text2)
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1_concatenated, text2_concatenated])
    return cosine_similarity(tfidf_matrix)[0][1]


# Define search function with related products and past searches
@app.get("/search")
async def search_products(query: str):
    query_stemmed = tokenize_and_stem(query)
    data['similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))

    # Filter products based on similarity
    results = data.sort_values(by=['similarity'], ascending=False).head(10)[['Title', 'Description', 'Category']]

    # Get the category of the most similar product
    top_category = data.loc[data['similarity'].idxmax(), 'Category']

    # Get related products from the same category (excluding the searched product)
    related_products = data[(data['Category'] == top_category) & (data['Title'] != query)]

    # Ensure we have at least three related products
    if len(related_products) < 3:
        # If there are not enough related products, supplement with additional products from the same category
        additional_products = data[(data['Category'] == top_category) & (data['Title'] != query)].drop_duplicates(
            subset=['Title']).head(3 - len(related_products))
        related_products = pd.concat([related_products, additional_products])

    # Shuffle the related products and return at least three
    if len(related_products) > 0:
        related_products = related_products.sample(n=min(3, len(related_products)), replace=False).head(3)[
            ['Title', 'Description', 'Category']]
    else:
        # If no related products, return an empty DataFrame
        related_products = pd.DataFrame(columns=['Title', 'Description', 'Category'])

    # Save user search to CSV file
    user_search = pd.DataFrame({'Query': [query]})
    user_search.to_csv('user_search_history.csv', mode='a', header=False, index=False)

    try:
        # Read past searches from CSV file
        past_searches = pd.read_csv('user_search_history.csv', names=['history recommendations'])
        if len(past_searches) >= 3:
            # If there are at least three past searches, return a random subset
            random_past_searches = past_searches.sample(n=3, replace=False)
            return JSONResponse(content={"Search Results": results.to_dict(orient='records'),
                                         "Related Products": related_products.to_dict(orient='records'),
                                         "Past User Searches": random_past_searches.to_dict(orient='records')})
        else:
            # If there are fewer than three past searches, return all available searches
            return JSONResponse(content={"Search Results": results.to_dict(orient='records'),
                                         "Related Products": related_products.to_dict(orient='records'),
                                         "Past User Searches": past_searches.to_dict(orient='records')})
    except FileNotFoundError:
        return HTTPException(status_code=404, detail="User search history file not found.")

# Define endpoint for past_searches
@app.get("/past_searches")
async def get_past_searches():
    try:
        # Read past searches from CSV file
        past_searches = pd.read_csv('user_search_history.csv', names=['history recommendations'])
        if len(past_searches) >= 3:
            # If there are at least three past searches, return a random subset
            random_past_searches = past_searches.sample(n=3, replace=False)
            return JSONResponse(content={"Past User Searches": random_past_searches.to_dict(orient='records')})
        else:
            # If there are fewer than three past searches, return all available searches
            return JSONResponse(content={"Past User Searches": past_searches.to_dict(orient='records')})
    except FileNotFoundError:
        return HTTPException(status_code=404, detail="User search history file not found.")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
