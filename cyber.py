import nltk
import math
import streamlit as st
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
tokenizer = PunktSentenceTokenizer()



# Initialize stemmer and stopwords
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

# Step 1: Sentence Tokenization
def split_into_sentences(text):
    return tokenizer.tokenize(text)

# Step 2: Create Frequency Matrix for each sentence
def create_frequency_matrix(sentences):
    freq_matrix = {}
    for sent in sentences:
        words = word_tokenize(sent.lower())
        words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words]
        freq_table = {}
        for word in words:
            freq_table[word] = freq_table.get(word, 0) + 1
        freq_matrix[sent] = freq_table
    return freq_matrix

# Step 3: Term Frequency (TF) calculation
def create_tf_matrix(freq_matrix):
    tf_matrix = {}
    for sent, freq_table in freq_matrix.items():
        tf_table = {}
        total_words = sum(freq_table.values())
        for word, count in freq_table.items():
            tf_table[word] = count / total_words
        tf_matrix[sent] = tf_table
    return tf_matrix

# Step 4: Document per word table
def create_documents_per_word(freq_matrix):
    word_doc_table = {}
    for freq_table in freq_matrix.values():
        for word in freq_table:
            word_doc_table[word] = word_doc_table.get(word, 0) + 1
    return word_doc_table

# Step 5: Inverse Document Frequency (IDF) calculation
def create_idf_matrix(freq_matrix, doc_per_words, total_docs):
    idf_matrix = {}
    for sent, freq_table in freq_matrix.items():
        idf_table = {}
        for word in freq_table:
            idf_table[word] = math.log10(total_docs / float(doc_per_words[word]))
        idf_matrix[sent] = idf_table
    return idf_matrix

# Step 6: TF-IDF Calculation
def create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    for sent, tf_table in tf_matrix.items():
        tfidf_table = {}
        for word, tf_val in tf_table.items():
            idf_val = idf_matrix[sent].get(word, 0)
            tfidf_table[word] = tf_val * idf_val
        tf_idf_matrix[sent] = tfidf_table
    return tf_idf_matrix

# Step 7: Score Sentences
def score_sentences(tfidf_matrix):
    sentence_scores = {}
    for sent, tfidf_table in tfidf_matrix.items():
        total_score = sum(tfidf_table.values())
        sentence_scores[sent] = total_score / len(tfidf_table) if tfidf_table else 0
    return sentence_scores

# Step 8: Average Score
def find_average_score(sentence_scores):
    return sum(sentence_scores.values()) / len(sentence_scores)

# Step 9: Generate Summary
def generate_summary(sentences, sentence_scores, threshold):
    summary = [sent for sent in sentences if sentence_scores.get(sent, 0) >= threshold]
    return " ".join(summary)

# Main summarizer
def summarize_text(text,threshold_ctr):
    sentences = split_into_sentences(text)
    freq_matrix = create_frequency_matrix(sentences)
    tf_matrix = create_tf_matrix(freq_matrix)
    doc_per_words = create_documents_per_word(freq_matrix)
    idf_matrix = create_idf_matrix(freq_matrix, doc_per_words, len(sentences))
    tfidf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)
    sentence_scores = score_sentences(tfidf_matrix)
    threshold = threshold_ctr
    print(threshold)
    summary = generate_summary(sentences, sentence_scores, threshold)
    return summary

# ----------------------- STREAMLIT APP -----------------------

st.markdown("Name : Prasad Jawadekar")
st.markdown("Bits id : 2024aa05482")
st.title("📝 Cyber security assignment 1 : Text Summarizer using TF-IDF")

st.markdown("Enter a long piece of text below and get a summary based on TF-IDF scoring.")

text_input = st.text_area("📌 Enter your text here:", height=250)

threshold_slider = st.slider(
    "🎯 Set summarization threshold (lower = longer summary)", 
    min_value=0.0, 
    max_value=0.09, 
    value=0.0,       # Default to auto threshold
    step=0.01
)


if st.button("🔍 Summarize"):
    if text_input.strip():
        

        result = summarize_text(text_input,threshold_slider)
        st.subheader("✅ Summary:")
        st.write("Length of input text:",len(text_input))
        st.write(" Length of result ",len(result))
        st.write()
        
        st.write(result)
    else:
        st.warning("Please enter some text to summarize.")
