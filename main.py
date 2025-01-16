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

def load_documents_from_directory(directory_path):
    """Load .txt documents into a list of dicts: [{"id": "...", "content": "..."}]."""
    documents = []
    doc_id = 0
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.lower().endswith(".txt"):
                full_path = os.path.join(root, filename)
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                documents.append({"id": f"doc_{doc_id}", "content": text})
                doc_id += 1
    return documents

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [t for t in tokens if t.isalpha() and t not in STOPWORDS]
    stemmed_tokens = [STEMMER.stem(tok) for tok in filtered_tokens]
    return stemmed_tokens

def build_inverted_index(documents):
    """Return {term: {doc_id: freq, ...}, ...}."""
    index_inverted = defaultdict(lambda: defaultdict(int))
    for doc in documents:
        doc_id = doc["id"]
        tokens = preprocess_text(doc["content"])
        freq = defaultdict(int)
        for t in tokens:
            freq[t] += 1
        for term, f in freq.items():
            index_inverted[term][doc_id] = f
    return index_inverted

def compute_tf(tf_scheme, raw_freq):
    """Compute TF according to user choice: b=Binary, n=Raw freq, l=Log freq."""
    if raw_freq == 0:
        return 0
    if tf_scheme == 'b':
        return 1
    elif tf_scheme == 'l':
        return 1 + math.log(raw_freq, 10)
    return raw_freq  # 'n' by default

def compute_idf(idf_scheme, df, total_docs):
    """Compute IDF according to user choice: n=None, t=Standard, p=Probabilistic."""
    if df == 0:
        return 0
    if idf_scheme == 'n':
        return 1.0
    elif idf_scheme == 'p':
        if df == total_docs:
            return 0
        return math.log((total_docs - df) / df, 10)
    return math.log(total_docs / df, 10)  # 't' by default

def build_weighted_index(inverted_index, total_docs, tf_scheme='n', idf_scheme='t'):
    weighted_index = defaultdict(dict)
    doc_vectors = defaultdict(dict)
    for term, postings in inverted_index.items():
        df = len(postings)
        idf_value = compute_idf(idf_scheme, df, total_docs)
        for doc_id, raw_freq in postings.items():
            tf_value = compute_tf(tf_scheme, raw_freq)
            weight = tf_value * idf_value
            weighted_index[term][doc_id] = weight
            doc_vectors[doc_id][term] = weight
    return weighted_index, doc_vectors

def similarity_cosine(q_vec, d_vec):
    """Cosine similarity."""
    dot_product = sum(q_vec[t] * d_vec.get(t, 0.0) for t in q_vec)
    q_norm = math.sqrt(sum(v*v for v in q_vec.values()))
    d_norm = math.sqrt(sum(v*v for v in d_vec.values()))
    return dot_product / (q_norm * d_norm) if q_norm and d_norm else 0.0

def similarity_jaccard(q_vec, d_vec):
    """Jaccard similarity (weighted)."""
    all_terms = set(q_vec.keys()) | set(d_vec.keys())
    numerator = sum(min(q_vec.get(t, 0.0), d_vec.get(t, 0.0)) for t in all_terms)
    denominator = sum(max(q_vec.get(t, 0.0), d_vec.get(t, 0.0)) for t in all_terms)
    return numerator / denominator if denominator else 0.0

def similarity_dice(q_vec, d_vec):
    """Dice similarity (weighted)."""
    common_terms = set(q_vec.keys()) & set(d_vec.keys())
    numerator = sum(min(q_vec[t], d_vec[t]) for t in common_terms)
    denominator = sum(q_vec.values()) + sum(d_vec.values())
    return (2.0 * numerator / denominator) if denominator else 0.0

def get_similarity_function(sim_name):
    if sim_name == 'jaccard':
        return similarity_jaccard
    elif sim_name == 'dice':
        return similarity_dice
    return similarity_cosine  # default

def search(query, weighted_index, doc_vectors, tf_scheme, idf_scheme, sim_name):
    query_tokens = preprocess_text(query)
    query_freq = defaultdict(int)
    for t in query_tokens:
        query_freq[t] += 1

    query_vec = {}
    total_docs = len(doc_vectors)
    for term, freq in query_freq.items():
        tf_value = compute_tf(tf_scheme, freq)
        df = len(weighted_index[term]) if term in weighted_index else 0
        idf_value = compute_idf(idf_scheme, df, total_docs)
        query_vec[term] = tf_value * idf_value

    sim_func = get_similarity_function(sim_name)
    scores = {}
    for doc_id, doc_vec in doc_vectors.items():
        scores[doc_id] = sim_func(query_vec, doc_vec)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def evaluate_metrics(results, relevant_docs, top_k=None):
    """Compute multiple metrics: global precision, recall, F1, P@K, R@K."""
    if not results:
        return {"precision": 0, "recall": 0, "f1": 0, "p@k": 0, "r@k": 0}
    
    # Global metrics (on entire results list)
    retrieved_docs = [doc_id for doc_id, _ in results]
    retrieved_set = set(retrieved_docs)
    relevant_retrieved = retrieved_set & relevant_docs

    precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
    recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    # At K
    if top_k and top_k > 0:
        top_k_results = results[:top_k]
        top_k_docs = [doc_id for doc_id, _ in top_k_results]
        top_k_relevant = set(top_k_docs) & relevant_docs
        p_at_k = len(top_k_relevant) / len(top_k_results)
        r_at_k = len(top_k_relevant) / len(relevant_docs) if relevant_docs else 0
    else:
        p_at_k, r_at_k = 0, 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "p@k": p_at_k,
        "r@k": r_at_k
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Search query.")
    parser.add_argument("--documents", type=str, default="./documents", help="Folder with .txt files.")
    parser.add_argument("--tf", type=str, default='n', choices=['b','n','l'], help="TF scheme.")
    parser.add_argument("--idf", type=str, default='t', choices=['n','t','p'], help="IDF scheme.")
    parser.add_argument("--sim", type=str, default='cosine', choices=['cosine','jaccard','dice'], help="Similarity.")
    parser.add_argument("--topk", type=int, default=5, help="Number of documents to display & evaluate (K).")
    args = parser.parse_args()

    print(f"Loading documents from: {args.documents}")
    docs = load_documents_from_directory(args.documents)
    print(f"Number of documents loaded: {len(docs)}")

    print("Building inverted index...")
    inverted_idx = build_inverted_index(docs)

    print(f"Building weighted index (TF='{args.tf}', IDF='{args.idf}')...")
    weighted_idx, doc_vectors = build_weighted_index(inverted_idx, len(docs),
                                                     tf_scheme=args.tf,
                                                     idf_scheme=args.idf)

    user_query = args.query
    relevant_docs = {"doc_3", "doc_5", "doc_7"} 
    results = search(user_query, weighted_idx, doc_vectors, args.tf, args.idf, args.sim)

    print(f"\nQuery: \"{user_query}\" (TF='{args.tf}', IDF='{args.idf}', SIM='{args.sim}')")
    print(f"Top {args.topk} results:")
    for doc_id, score in results[:args.topk]:
        print(f"  {doc_id} => {score:.4f}")

    metrics = evaluate_metrics(results, relevant_docs, top_k=args.topk)
    print("\n--- Evaluation Metrics ---")
    print(f"Global Precision: {metrics['precision']:.2f}")
    print(f"Global Recall:    {metrics['recall']:.2f}")
    print(f"Global F1:        {metrics['f1']:.2f}")
    print(f"P@{args.topk}:           {metrics['p@k']:.2f}")
    print(f"R@{args.topk}:           {metrics['r@k']:.2f}")
