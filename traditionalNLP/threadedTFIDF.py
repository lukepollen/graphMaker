### Imports ###

import itertools
import nltk
import numpy as np
import os
import pandas as pd
import pickle

from concurrent.futures import ThreadPoolExecutor
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
#from tqdm import tqdm

# Download necessary resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

### End of Imports ###


### Function Definition ###

# Function to compute TF-IDF matrix and feature names
def compute_tfidf(page_texts):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 3), tokenizer=custom_tokenizer)
    tfidf_matrix = vectorizer.fit_transform(page_texts)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names


# Function to compute POS tags for a given text
def compute_pos_tags(text):

    # Check whether posCount is a globalvar. Set if not.
    if 'posCount' not in globals():
        global posCount
        posCount = 0

    # Create tags for current text
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    # Increment global counter and print alert for every 10,000 texts parsed
    posCount = posCount + 1
    if (posCount % 10000 == 0):
        print('Processed row ' + str(posCount) + ", and added Parts of Speech tags to it")

    return pos_tags


# Function to get top N similar pages and their similarity scores
def top_n_similar_pages(tfidf_matrix, target_page_index, n):

    cosine_similarities = cosine_similarity(tfidf_matrix[target_page_index], tfidf_matrix)
    most_similar_page_indices = cosine_similarities.argsort().flatten()[-n-1:-1][::-1]
    most_similar_page_scores = cosine_similarities[0, most_similar_page_indices]
    return most_similar_page_indices, most_similar_page_scores


# Function tp strips URLs to their base
def clean_page(page):
    cleaned_page = page.split('?', 1)[0].split('#', 1)[0]
    return cleaned_page


def single_custom_tokenizer(text):
    num_cores = max(os.cpu_count() - 1, 1)
    return parallel_custom_tokenizer([text], num_threads=num_cores, disable_tqdm=True)[0]


def parallel_custom_tokenizer(texts, num_threads, disable_tqdm=False):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        tokenized_texts = list(tqdm(executor.map(custom_tokenizer, texts),
                                    total=len(texts),
                                    desc="Tokenizing",
                                    position=0,
                                    leave=False,
                                    disable=disable_tqdm))
    return tokenized_texts


def custom_tokenizer(text):

    # Reference previously set pbar so that we can update the progress bar as we tokenize
    global pbar

    # Filter tags. Exclude tags with undesired starts and non noun ending condition
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    excluded_start_tags = {'VB', 'SP', 'RB', 'Co'}
    required_end_tag = 'NN'
    filtered_terms = [word for word, tag in pos_tags if tag not in excluded_start_tags]
    n_grams = []

    # Create ngrams
    for n in range(1, 4):
        for i in range(len(filtered_terms) - n + 1):
            n_gram = ' '.join(filtered_terms[i:i + n])
            n_grams.append(n_gram)

    # Filter the ngrams based on the tags and number of cardinal numbers in the tag
    n_grams = [n_gram for n_gram in n_grams if pos_tag(n_gram.split())[-1][1].endswith(required_end_tag)]
    cd_count = sum(1 for word, tag in pos_tags if tag == 'CD')
    n_grams = [n_gram for n_gram in n_grams if len(pos_tag(n_gram.split())) <= cd_count * 2]
    pbar.update(1)

    # Append to the global structure for n_grams
    global global_ngrams
    global_ngrams.append(n_grams)

    # Return the generated tags
    return n_grams


def retain_terms(vectorizer, terms_to_retain):

    # Get the original vocabulary and IDF values
    original_vocab = vectorizer.get_feature_names_out()
    original_idf = vectorizer.idf_

    # Filter the vocabulary based on the terms to retain
    filtered_vocab = []
    filtered_indices = []
    for idx, term in enumerate(original_vocab):
        if term in terms_to_retain:
            filtered_vocab.append(term)
            filtered_indices.append(idx)

    # Create a new vectorizer with the filtered vocabulary
    filtered_vectorizer = TfidfVectorizer(vocabulary=filtered_vocab)

    # Compute the filtered IDF values based on the filtered indices
    filtered_idf = np.zeros(len(filtered_vocab))
    for i, idx in enumerate(filtered_indices):
        filtered_idf[i] = original_idf[idx]

    # Set the filtered IDF values in the new vectorizer
    filtered_vectorizer.idf_ = filtered_idf

    return filtered_vectorizer


def retain_terms_with_progress(vectorizer, terms_to_retain):

    # Get the original vocabulary and IDF values
    original_vocab = np.array(vectorizer.get_feature_names_out())
    original_idf = vectorizer.idf_
    print('Got original vocab and idf values.')

    # Create a boolean mask for the terms to retain
    #retain_mask = np.isin(original_vocab, terms_to_retain)
    terms_to_retain_set = set(terms_to_retain)
    retain_mask = np.array([term in terms_to_retain_set for term in original_vocab])

    print('Masked the terms to retain.')

    # Filter the vocabulary based on the terms to retain
    filtered_vocab = original_vocab[retain_mask]
    #filtered_indices = np.where(retain_mask)[0]
    print('Filtered vocab to terms to retain')

    # Create a new vectorizer with the filtered vocabulary
    filtered_vectorizer = TfidfVectorizer(vocabulary=filtered_vocab)
    print('Created a new vectorizer with the reduced vocab')

    # Compute the filtered IDF values based on the filtered indices
    filtered_idf = original_idf[retain_mask]
    print('Filtered idf values of original vectorizer')

    # Set the filtered IDF values in the new vectorizer
    filtered_vectorizer.idf_ = filtered_idf
    print('Fitted new vectorizer with the filtered idf values')
    print('Returning the new vectorizer')

    return filtered_vectorizer


def create_tfidf_dataframe(pages, tfidf_matrix, feature_names):
    """
    Create a TF-IDF dataframe from the given data.

    Args:
        pages (list): List of page names.
        tfidf_matrix (scipy.sparse.coo_matrix): Sparse TF-IDF matrix.
        feature_names (list): List of feature names.

    Returns:
        pandas.DataFrame: TF-IDF dataframe with columns ['Page', 'Page Text', 'TF-IDF Term', 'TF-IDF Score'].
    """

    # Create a progress bar
    progress_bar = tqdm(total=tfidf_matrix.nnz)

    # Get the non-zero indices and scores from the sparse matrix
    nonzero_indices = tfidf_matrix.nonzero()
    tfidf_scores = tfidf_matrix.data

    # Get the page indices
    page_indices = nonzero_indices[0]
    term_indices = nonzero_indices[1]

    # Retrieve the corresponding pages and page texts
    pages = np.array(pages)
    terms = np.array(feature_names)

    # Initialize the list to store TF-IDF data
    tfidf_data = []

    # Generate rows in the form of page, page text, term for page, score for term on page
    for page_index, term_index, score in zip(page_indices, term_indices, tfidf_scores):
        page = pages[page_index]
        term = terms[term_index]

        tfidf_data.append((page, term, score))

        # Update the progress bar after processing each term
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    # Create a pandas DataFrame from the TF-IDF data
    tfidf_df = pd.DataFrame(tfidf_data, columns=['Page', 'TF-IDF Term', 'TF-IDF Score'])

    # Save the first 10,000 rows to an Excel file
    tfidf_df_subset = tfidf_df.head(10000)
    tfidf_df_subset.to_excel('tfidf_subset.xlsx', index=False)

    # Pickle the TF-IDF dataframe
    #with open('tfidf_dataframe.pickle', 'wb') as f:
    #    pickle.dump(tfidf_df, f)

    return tfidf_df


### End of Function Definition ###


### Frozen Variables ###

# Create credentials object which will be used in retrieving data for tfidf
credentials = ''

# SQL query to execute against bigquery table
query = """
WITH latest_page_text AS (
    SELECT Page,
           `Page Text`,
           Date,
           ROW_NUMBER() OVER (PARTITION BY Page ORDER BY Date DESC) AS row_num
    FROM `yourdataset.yourtable`
    WHERE LENGTH(`Page Text`) > 0
)
SELECT Page, `Page Text`
FROM latest_page_text
WHERE row_num = 1;
"""

# Number of similar pages to return, based on the TFIDF matrix
n_similar = 5

### End of Frozen Variables


### Dynamic Logic ###

print('\n')
print('Fetching and cleaning data...')
# Execute the query and store the results in a pandas DataFrame
df = pd.read_gbq(query, project_id=credentials.project_id, credentials=credentials)

# Apply the cleaning function to the Page column
df['Page'] = df['Page'].apply(clean_page)

# Remove duplicates from the 'Page' column
df = df.drop_duplicates(subset='Page')

# Convert the dataframe columns to list items for use in TFIDF and similarity calculations
pages = df['Page'].tolist()
page_texts = df['Page Text'].tolist()
print('Data fetch and clean complete')

# Create container for the n-grams the tokenizer with generate
global_ngrams = []


# If we previously created the vectorizer and tfidf_matrix, load them
try:
    with open('vectorizer.pickle', 'rb') as file:

        # Attempt to load the vectorizer, the tfidf_matrix, and feature names
        vectorizer = pickle.load(file)

# If we failed to load the vectorizer, create it
except FileNotFoundError as fnfe:

    # If not vectorizer exists as a pickle, create it
    vectorizer = TfidfVectorizer(stop_words='english',
                                 max_df=0.7,
                                 ngram_range=(1, 3),
                                 tokenizer=single_custom_tokenizer,
                                 preprocessor=None)


# Try to load the feature names, from previous run of this script.
try:

    # Load feature_names returned by our vectorizer over the page_texts for computing tfidf dataframe.
    with open('featureNames.pickle', 'rb') as file:
        features_names = pickle.load(file)

# If we could not load the tfidf_matrix and feature names from previous run, create them with the vectorizer.
except FileNotFoundError as fnfe:

    # If File is not found, warn of the error.
    print(str(fnfe))

    # Set a global variable which can be accessed within the tokenising function.
    global pbar
    pbar = tqdm(total=len(page_texts), desc="Tokenizing", position=0, leave=False)

    # Fit the vectorizer on the page_texts.
    vectorizer.fit(page_texts)

    # Flatten the list of lists, containing the ngrams of interest, into a single list.
    flattened_ngrams = list(itertools.chain.from_iterable(global_ngrams))

    # Get unique feature_names.
    feature_names = list(set(flattened_ngrams))

    # Save the feature names.
    feature_names_path = 'featureNames.pickle'
    with open(feature_names_path, 'wb') as file:
        # Save the data using pickle.
        pickle.dump(feature_names, file)

    # Close the display.
    pbar.close()


# Try to load a normalised tfidf Matrix
try:
    with open('normalisedTfidfMatrix.pickle', 'rb') as file:
        normalized_tfidf_matrix = pickle.load(file)

# If load failed, create the normalised tfidf matrix
except FileNotFoundError as fnfe:

    # Set the custom n-grams as vocabulary in the vectorizer.
    #vectorizer.vocabulary_ = {term: idx for idx, term in enumerate(feature_names)}

    # Filter the idf values and terms down to the subset of desired features and the scores for these features
    vectorizer = retain_terms_with_progress(vectorizer, feature_names)

    # # Save the original vectorizer object.
    # vector_path = 'vectorizer.pickle'
    # with open(vector_path, 'wb') as file:
    #     # Save the data using pickle.
    #     pickle.dump(vectorizer, file)

    # Compute the TF-IDF matrix for the selected n-grams.
    tfidf_matrix = vectorizer.transform(page_texts) #.toarray()

    # Save the tfidf_matrix.
    matrix_path = 'tfidfMatrix.pickle'
    with open(matrix_path, 'wb') as file:
        # Save the data using pickle.
        pickle.dump(tfidf_matrix, file)

    # Notmalise the tfidf matrix
    #transformer = TfidfTransformer()
    #normalized_tfidf_matrix = transformer.fit_transform(tfidf_matrix).toarray()

    transformer = TfidfTransformer()
    normalized_tfidf_matrix = transformer.fit_transform(tfidf_matrix)
    normalized_tfidf_matrix_sparse = csr_matrix(normalized_tfidf_matrix)
    #normalized_tfidf_matrix = transformer.transform(tfidf_matrix)

    # Save the tfidf_matrix.
    matrix_path = 'normalisedTfidfMatrixSparse.pickle'
    with open(matrix_path, 'wb') as file:
        # Save the data using pickle.
        pickle.dump(normalized_tfidf_matrix, file)

matrix_path = 'normalisedTfidfMatrixSparse.pickle'
with open(matrix_path, 'rb') as file:
    # Save the data using pickle.
    normalized_tfidf_matrix_sparse = pickle.load(file)

# Assign the tfidf dataFrame
tfidf_df = create_tfidf_dataframe(pages, page_texts, normalized_tfidf_matrix_sparse, features_names)
#columns = ['Page', 'TF-IDF Term', 'TF-IDF Score']
#tfidf_df.columns = columns
print('Created TFIDF dataframe!')

# Apply the function to compute POS tags for the "Page Text" column
print('Applying final POS Tags')
print('Length of dataframe is ' + str(len(tfidf_df['TF-IDF Term'])))
tfidf_df['POS Tags'] = tfidf_df['TF-IDF Term'].apply(compute_pos_tags)




#with open('tfidfDataframe.pickle', 'rb') as file:
#    tfidf_df = pickle.load(file)


# Convert the dataframe to a NumPy array
array = tfidf_df.to_numpy()

# Save the NumPy array
np.save('tfidf_dataframe.npy', array)

# Load the NumPy array
loaded_array = np.load('tfidf_dataframe.npy')

# Convert the loaded array back to a dataframe
loaded_df = pd.DataFrame(loaded_array, columns=['Page', 'TF-IDF Term', 'TF-IDF Score', 'POS Tags'])


# Some final exclusion


# Some final save
#dataframe_path = 'tfidfDataframe.pickle'
#with open(dataframe_path, 'wb') as file:
#    # Save the data using pickle.
#    pickle.dump(tfidf_df, file)

# # Apply the function to compute POS tags for the "Page Text" column
# print('Applying final POS Tags')
# print('Length of dataframe is ' + str(len(tfidf_df['Page Text'])))
# tfidf_df['POS Tags'] = tfidf_df['Page Text'].apply(compute_pos_tags)


### ERRATA ###

# # Create a progress bar so we can monitor progress of row creation for the tf_idf dataFrame. Stored in tfidf_data.
# progress_bar = tqdm(total=len(pages) * len(feature_names))
#
# # Generate rows in the form of page, page text, term for page, score for term on page.
# for i, (page, page_text) in enumerate(zip(pages, page_texts)):
#     row = tfidf_matrix[i]
#     nonzero_indices = row.nonzero()[1]
#     tfidf_scores = row.data
#
#     # Get term scores for the current page
#     for index, score in zip(nonzero_indices, tfidf_scores):
#         term = feature_names[index]
#         tfidf_data.append((page, page_text, term, score))
#
#         # Update the progress bar after processing each term
#         progress_bar.update(1)
#
# # Close the progress bar
# progress_bar.close()




# # Custom tokenizer function
# def custom_tokenizer(text):
#
#     words = word_tokenize(text)
#     pos_tags = pos_tag(words)
#     excluded_start_tags = {'VB', 'SP', 'RB', 'Co'}
#     required_end_tag = 'NN'
#     filtered_terms = [word for word, tag in pos_tags if tag not in excluded_start_tags]
#     n_grams = []
#
#     for n in range(1, 4):
#         for i in range(len(filtered_terms) - n + 1):
#             n_gram = ' '.join(filtered_terms[i:i + n])
#             n_grams.append(n_gram)
#
#     n_grams = [n_gram for n_gram in n_grams if pos_tag(n_gram.split())[-1][1].endswith(required_end_tag)]
#     cd_count = sum(1 for word, tag in pos_tags if tag == 'CD')
#     n_grams = [n_gram for n_gram in n_grams if len(pos_tag(n_gram.split())) <= cd_count * 3]
#
#     return n_grams


### End of Errata - GNU Terry Pratchett ###


























