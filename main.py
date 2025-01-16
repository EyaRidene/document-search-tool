import os
import math
import nltk
import argparse
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

LANGUAGE = "english"  # or "french", etc.
STOPWORDS = set(stopwords.words(LANGUAGE))
STEMMER = SnowballStemmer(LANGUAGE)

# -------------------------------------------------
# 1. Read documents
# -------------------------------------------------

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

# -------------------------------------------------
# 2. Preprocessing
# -------------------------------------------------

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

# -------------------------------------------------
# 3. Build Inverted Index
# -------------------------------------------------

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

# -------------------------------------------------
# 4. Weighting (TF & IDF)
# -------------------------------------------------

def compute_tf(tf_scheme, raw_freq):
    """
    Computes the TF according to user choice (tf_scheme).
    tf_scheme can be:
      - 'b': binary
      - 'n': raw frequency
      - 'l': log frequency
    """
    if raw_freq == 0:
        return 0
    if tf_scheme == 'b':
        return 1
    elif tf_scheme == 'l':
        return 1 + math.log(raw_freq, 10)
    # default: 'n'
    return raw_freq

def compute_idf(idf_scheme, df, total_docs):
    """
    Computes the IDF according to user choice (idf_scheme).
    idf_scheme can be:
      - 'n': none (always 1)
      - 't': standard ( log((N+1)/df) )
      - 'p': probabilistic ( log((N - df) / df) )
    """
    if df == 0:
        return 0

    if idf_scheme == 'n':
        return 1.0
    elif idf_scheme == 'p':
        # Avoid division by zero:
        if df == total_docs:
            # log((N-df)/df) would be log(0), we keep it minimal
            return 0
        return math.log((total_docs - df) / df, 10)
    # default: 't'
    # log((N+1)/df)
    return math.log((total_docs + 1) / df, 10)

def build_weighted_index(inverted_index, total_docs, tf_scheme='n', idf_scheme='t'):
    """
    Returns two structures:
    1) weighted_index = { term: { doc_id: tf*idf }, ... }
    2) doc_vectors = { doc_id: { term: tf*idf, ... }, ... } (for easier searching)
    
    We compute the TF-IDF (or variations) as (TF * IDF).
    """
    weighted_index = defaultdict(dict)
    doc_vectors = defaultdict(dict)

    for term, postings in inverted_index.items():
        df = len(postings)
        idf_value = compute_idf(idf_scheme, df, total_docs)

        for doc_id, tf in postings.items():
            tf_idf_value = tf * idf_value
            weighted_index[term][doc_id] = tf_idf_value
            doc_vectors[doc_id][term] = tf_idf_value

    return weighted_index, doc_vectors

# -------------------------------------------------
# 5. Similarity Functions
# -------------------------------------------------

def similarity_cosine(q_vec, d_vec):
    """
    Cosine similarity = dot(q,d) / (||q|| * ||d||).
    """
    # Dot product
    dot_product = 0.0
    for term, q_weight in q_vec.items():
        d_weight = d_vec.get(term, 0.0)
        dot_product += q_weight * d_weight
    
    # Norms
    q_norm = math.sqrt(sum(v**2 for v in q_vec.values()))
    d_norm = math.sqrt(sum(v**2 for v in d_vec.values()))
    
    if q_norm == 0 or d_norm == 0:
        return 0.0
    
    return dot_product / (q_norm * d_norm)

def similarity_jaccard(q_vec, d_vec):
    """
    Jaccard similarity = |Q ∩ D| / |Q ∪ D|.
    In a weighted context, a typical adaptation is:
       sum(min(q_vec[term], d_vec[term])) / sum(max(q_vec[term], d_vec[term])).
    """
    all_terms = set(q_vec.keys()) | set(d_vec.keys())
    numerator = 0.0
    denominator = 0.0

    for term in all_terms:
        q_w = q_vec.get(term, 0.0)
        d_w = d_vec.get(term, 0.0)
        numerator += min(q_w, d_w)
        denominator += max(q_w, d_w)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def similarity_dice(q_vec, d_vec):
    """
    Dice similarity = (2 * |Q ∩ D|) / (|Q| + |D|).
    Weighted adaptation:
       2 * sum(min(q_vec[term], d_vec[term])) / (sum(q_vec.values()) + sum(d_vec.values())).
    """
    numerator = 0.0
    for term in set(q_vec.keys()) & set(d_vec.keys()):
        numerator += min(q_vec[term], d_vec[term])
    denominator = sum(q_vec.values()) + sum(d_vec.values())

    if denominator == 0:
        return 0.0
    
    return (2.0 * numerator) / denominator

def get_similarity_function(sim_name):
    """
    Returns the appropriate similarity function given sim_name.
    """
    if sim_name == 'jaccard':
        return similarity_jaccard
    elif sim_name == 'dice':
        return similarity_dice
    return similarity_cosine

# -------------------------------------------------
# 6. Apply Search
# -------------------------------------------------

def search(query, weighted_index, doc_vectors, tf_scheme, idf_scheme, sim_name):
    """
    Steps:
    - Preprocess the 'query' string
    - Build a query TF/IDF vector based on the chosen scheme
    - Calculate similarity with each doc (based on the chosen similarity metric)
    - Return sorted doc_ids with their similarity scores
    """
    # 1) Preprocess query
    query_tokens = preprocess_text(query)

    # 2) Count frequencies in the query
    query_frequencies = defaultdict(int)
    for t in query_tokens:
        query_frequencies[t] += 1

    # 3) Build a vector for the query
    query_vector = {}
    total_docs = len(doc_vectors)
    
    # For each term in the query, compute TF * IDF according to chosen scheme
    for term, freq in query_frequencies.items():
        # compute TF for query
        tf_value = compute_tf(tf_scheme, freq)
        
        # compute IDF (if the term is known in the index)
        if term in weighted_index:
            df = len(weighted_index[term])
        else:
            df = 0
        
        idf_value = compute_idf(idf_scheme, df, total_docs)
        
        query_vector[term] = tf_value * idf_value

    # 4) Get the chosen similarity function
    similarity_func = get_similarity_function(sim_name)

    # 5) Calculate similarities with each document
    scores = {}
    for doc_id, doc_vec in doc_vectors.items():
        score = similarity_func(query_vector, doc_vec)
        scores[doc_id] = score

    # 6) Sort by score descending
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

# -------------------------------------------------
# 7. Evaluate
# -------------------------------------------------

def evaluate_precision_recall(results, relevant_docs):
    """
    Calculate precision and recall based on the retrieved results and relevant documents.
    
    Args:
        results (list of tuples): Ranked list of results [(doc_id, score), ...].
        relevant_docs (set): Set of relevant document IDs for the query.
    
    Returns:
        tuple: (precision, recall)
    """
    if not results:
        return 0.0, 0.0

    retrieved_docs = [doc_id for doc_id, _ in results]
    retrieved_set = set(retrieved_docs)
    relevant_retrieved = retrieved_set & relevant_docs

    precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
    recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0

    return precision, recall

# -------------------------------------------------
# 8. Main Demo
# -------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search and Evaluate Precision and Recall.")
    parser.add_argument("--query", type=str, required=True, help="The search query string.")
    parser.add_argument("--documents", type=str, default="./documents", 
                        help="Folder containing .txt documents.")
    parser.add_argument("--tf", type=str, default='n', choices=['b','n','l'],
                        help="TF weighting scheme: b=Binary, n=Raw freq, l=Log freq.")
    parser.add_argument("--idf", type=str, default='t', choices=['n','t','p'],
                        help="IDF weighting scheme: n=None, t=Standard, p=Probabilistic.")
    parser.add_argument("--sim", type=str, default='cosine', choices=['cosine','jaccard','dice'],
                        help="Similarity measure.")
    args = parser.parse_args()

    # 1) Load documents
    print(f"Loading documents from: {args.documents}")
    docs = load_documents_from_directory(args.documents)
    print(f"Number of documents loaded: {len(docs)}")

    # 2) Build inverted index
    print("Building inverted index...")
    inverted_idx = build_inverted_index(docs)

    # 3) Build Weighted Index (TF x IDF)
    print(f"Building weighted index using TF='{args.tf}', IDF='{args.idf}'")
    N = len(docs)
    weighted_idx, doc_vectors = build_weighted_index(inverted_idx, N, 
                                                     tf_scheme=args.tf, 
                                                     idf_scheme=args.idf)

    # 4) Process query
    user_query = args.query
    relevant_docs = {"doc_3", "doc_5", "doc_7"}
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
