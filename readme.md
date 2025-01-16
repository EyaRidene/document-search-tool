# README – Basic TF-IDF Search Engine in Python

This project demonstrates a **simple information retrieval system** using Python and NLTK (Natural Language Toolkit). It does the following:

1. Loads and scans `.txt` files from a specified directory.
2. Preprocesses text (tokenization, stopword removal, stemming).
3. Builds an **inverted index** that maps each term to the documents in which it appears.
4. Computes **TF-IDF** weights for each term–document pair.
5. Performs **cosine similarity**–based ranking of documents given a user query.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [How It Works](#how-it-works)
   1. [1. Reading Documents](#1-reading-documents)
   2. [2. Preprocessing](#2-preprocessing)
   3. [3. Building the Inverted Index](#3-building-the-inverted-index)
   4. [4. Computing TF-IDF](#4-computing-tf-idf)
   5. [5. Cosine Similarity and Searching](#5-cosine-similarity-and-searching)
   6. [6. Main Demo](#6-main-demo)
4. [Usage](#usage)
5. [Customization](#customization)
6. [Possible Improvements](#possible-improvements)
7. [License](#license)

---

## Prerequisites

- **Python 3.7+** (earlier versions may work, but are not tested)
- **NLTK** (Natural Language Toolkit). To install it:

  ```bash
  pip install nltk
  ```

---

Additionally, download the necessary NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## Project Structure

```plaintext
project-folder/
├── documents/               # Directory containing .txt files
├── main.py                  # Main script
├── README.md                # Documentation
```

---

# TF-IDF Search Engine Project

This project is a Python-based search engine that allows users to query a set of documents using TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity. Below is a detailed description of the implementation.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [How It Works](#how-it-works)
   1. [1. Reading Documents](#1-reading-documents)
   2. [2. Preprocessing](#2-preprocessing)
   3. [3. Building the Inverted Index](#3-building-the-inverted-index)
   4. [4. Computing TF-IDF](#4-computing-tf-idf)
   5. [5. Cosine Similarity and Searching](#5-cosine-similarity-and-searching)
   6. [6. Main Demo](#6-main-demo)
4. [Usage](#usage)
5. [Customization](#customization)
6. [Possible Improvements](#possible-improvements)
7. [License](#license)

---

## Prerequisites

Ensure the following Python libraries are installed:

- `nltk`
- `os` (standard library)
- `math` (standard library)
- `collections` (standard library)

Additionally, download the necessary NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## Project Structure

```plaintext
project-folder/
├── documents/               # Directory containing .txt files
├── main.py                  # Main script
├── README.md                # Documentation
```

---

## How It Works

### 1. Reading Documents

- Recursively scans a directory for `.txt` files.
- Loads their content into a structured list of documents with unique IDs.

### 2. Preprocessing

- Converts text to lowercase.
- Tokenizes text using NLTK.
- Removes stopwords and non-alphabetic tokens.
- Applies stemming to reduce words to their base forms.

### 3. Building the Inverted Index

- Maps terms to the documents in which they appear.
- Tracks the frequency of each term in each document.

### 4. Computing TF-IDF

- Calculates Term Frequency (TF) for each term in each document.
- Calculates Inverse Document Frequency (IDF) for each term.
- Combines TF and IDF to compute the TF-IDF score.

### 5. Cosine Similarity and Searching

- Builds a query vector using the same preprocessing and TF-IDF calculation as for documents.
- Computes the cosine similarity between the query vector and document vectors.
- Ranks documents based on similarity scores.

---

## Usage

1. Place your `.txt` files in the `documents/` directory.
2. Run the script:

```bash
python main.py
```

3. Enter a query when prompted.

4. Sample Output :

```plaintext
Loading documents from: ./documents
Number of documents loaded: 5
Building inverted index...
Computing TF-IDF...

Searching for: "example query"
```

```bash
#1 | Document ID: 12 | Score: 0.4587
#2 | Document ID: 4  | Score: 0.3412
#3 | Document ID: 19 | Score: 0.2975
...
```

---

## Customization

- **Stopwords Language**: Change the `LANGUAGE` variable to use a different stopwords list (e.g., "french").
- **Stemming**: Swap out the `SnowballStemmer` for a different stemming algorithm if needed.
- **Directory Path**: Modify the `folder_path` variable in `main.py` to point to your desired directory.

---

## Possible Improvements

- Add support for multiple file formats (e.g., `.pdf`, `.docx`).
- Implement query expansion techniques to improve search results.
- Use advanced NLP techniques such as lemmatization instead of stemming.
- Optimize performance for large document sets using parallel processing.

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code as long as proper attribution is given.
