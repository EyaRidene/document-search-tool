import os
import math
import nltk
import argparse
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

LANGUAGE = "english"  # "french", "english", etc.
STOPWORDS = set(stopwords.words(LANGUAGE))

STEMMER = SnowballStemmer(LANGUAGE)

######################################
# 1. Read documents
######################################

def load_documents_from_directory(directory_path):
    """
    Recursively scans 'directory_path' for .txt files and loads them into
    a list of dicts: [{"id": "...", "content": "..."}].
    """
    documents = []
    doc_id = 0
    
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.lower().endswith(".txt"):
                full_path = os.path.join(root, filename)
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                documents.append({
                    "id": f"doc_{doc_id}",
                    "content": text
                })
                doc_id += 1
    return documents

######################################
# 2. Preprocessing
######################################

def preprocess_text(text):
    """
    - Lowercase
    - Tokenize (NLTK's word_tokenize)
    - Remove stopwords and non-alphabetic tokens
    - Stem (SnowballStemmer)
    """
    # 1) Lowercase
    text = text.lower()

    # 2) Tokenize
    tokens = nltk.word_tokenize(text)

    # 3) Remove stopwords and non-alphabetic tokens
    filtered_tokens = [
        t for t in tokens
        if t.isalpha() and t not in STOPWORDS
    ]

    # 4) Stemming
    stemmed_tokens = [STEMMER.stem(tok) for tok in filtered_tokens]

    return stemmed_tokens

######################################
# 3. Build Inverted Index
######################################

def build_inverted_index(documents):
    """
    Creates an inverted index:
      { term: { doc_id: freq, ... }, ... }
    where freq is the number of occurrences of 'term' in document 'doc_id'.
    """
    index_inverted = defaultdict(lambda: defaultdict(int))

    for doc in documents:
        doc_id = doc["id"]
        content = doc["content"]

        # Preprocess
        tokens = preprocess_text(content)

        # Count frequencies in this doc
        frequencies = defaultdict(int)
        for t in tokens:
            frequencies[t] += 1

        # Update inverted index
        for term, freq in frequencies.items():
            index_inverted[term][doc_id] = freq

    return index_inverted

######################################
# 4. Compute TF-IDF
######################################

def compute_tf_idf(inverted_index, total_docs):
    """
    Returns two structures:
    1) tf_idf_index = { term: { doc_id: tf_idf_value }, ... }
    2) doc_vectors = { doc_id: { term: tf_idf_value, ... }, ... } (for easier searching)
    """
    tf_idf_index = defaultdict(dict)
    doc_vectors = defaultdict(dict)

    for term, postings in inverted_index.items():
        df = len(postings)
        idf = math.log((total_docs + 1) / df)

        for doc_id, tf in postings.items():
            tf_idf_value = tf * idf
            tf_idf_index[term][doc_id] = tf_idf_value
            doc_vectors[doc_id][term] = tf_idf_value

    return tf_idf_index, doc_vectors

######################################
# 5. Cosine Similarity Search
######################################

def cosine_similarity(q_vec, d_vec):
    # 1) dot product
    dot = 0.0
    for term, q_weight in q_vec.items():
        d_weight = d_vec.get(term, 0.0)
        dot += q_weight * d_weight
    
    # 2) norms
    q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
    d_norm = math.sqrt(sum(v * v for v in d_vec.values()))
    
    # 3) avoid division by zero
    if q_norm == 0 or d_norm == 0:
        return 0.0
    
    return dot / (q_norm * d_norm)

######################################
# 6. Apply Search
######################################

def search(query, tf_idf_index, doc_vectors):
    """
    - Preprocess the 'query' string
    - Build a TF-IDF vector for the query
    - Calculate cosine similarity with each doc
    - Return sorted doc_ids with their similarity scores
    """
    # 1) Preprocess query
    query_tokens = preprocess_text(query)

    # 2) Count frequencies in the query
    query_frequencies = defaultdict(int)
    for t in query_tokens:
        query_frequencies[t] += 1

    # 3) Build a TF-IDF vector for the query
    query_vector = {}
    total_docs = len(doc_vectors)
    for term, freq in query_frequencies.items():
        if term in tf_idf_index:
            df = len(tf_idf_index[term])
            if df > 0:
                idf = math.log((total_docs + 1) / df)
                query_vector[term] = freq * idf
        else:
            query_vector[term] = 0.0

    # 5) Calculate similarities with each document
    scores = {}
    for doc_id, doc_vec in doc_vectors.items():
        score = cosine_similarity(query_vector, doc_vec)
        scores[doc_id] = score

    # 6) Sort by score descending
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

######################################
# 7. Evaluate
######################################

def evaluate_precision_recall(results, relevant_docs):
    """
    Calculate precision and recall based on the retrieved results and relevant documents.
    
    Args:
        results (list of tuples): Ranked list of results [(doc_id, score), ...].
        relevant_docs (set): Set of relevant document IDs for the query.
    
    Returns:
        tuple: Precision and Recall as floats.
    """
    if not results:
        return 0.0, 0.0

    retrieved_docs = [doc_id for doc_id, _ in results]
    retrieved_set = set(retrieved_docs)
    relevant_retrieved = retrieved_set & relevant_docs

    precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
    recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0

    return precision, recall

######################################
# 8. Main Demo
######################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search and Evaluate Precision and Recall.")
    parser.add_argument("query", type=str, help="The search query string.")
    args = parser.parse_args()

    folder_path = "./documents"

    # 1) Load documents
    print(f"Loading documents from: {folder_path}")
    docs = load_documents_from_directory(folder_path)
    print(f"Number of documents loaded: {len(docs)}")

    # 2) Build inverted index
    print("Building inverted index...")
    inverted_idx = build_inverted_index(docs)

    # 3) Compute TF-IDF
    print("Computing TF-IDF...")
    N = len(docs)
    tf_idf_idx, doc_vectors = compute_tf_idf(inverted_idx, N)

    # 4) Process query from terminal
    user_query = args.query
    relevant_docs = {"doc_6", "doc_71", "doc_41"}
    print(f"\nSearching for: \"{user_query}\"")
    results = search(user_query, tf_idf_idx, doc_vectors)

    # Print top 5 results
    print("Top 5 results:")
    for doc_id, score in results[:5]:
        print(f"  DocID={doc_id}, Score={score:.4f}")

    print("Results:", results)
    print("Relevant Docs:", relevant_docs)

    # Evaluate the results
    precision, recall = evaluate_precision_recall(results[:5], relevant_docs)
    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")