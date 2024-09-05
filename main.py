import pandas as pd
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Reshape
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tensorflow.keras.optimizers import Adam

# Check if GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# Load datasets from CSV files
sensitive_terms = pd.read_csv('sensitive_terms.csv')
cover_queries = pd.read_csv('cover_queries.csv')
generated_cover_queries = pd.read_csv('generated_cover_queries.csv')
query_results = pd.read_csv('query_results.csv')
clickstream = pd.read_csv('clickstream.csv')

def build_neural_network(input_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128))
    model.add(LSTM(128))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(input_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Encode the cover queries
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(cover_queries['query'])
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(onehot_encoded, onehot_encoded, test_size=0.2)

# Train a simple neural network
model = build_neural_network(input_dim=onehot_encoded.shape[1])
model.fit(X_train, y_train, epochs=10, batch_size=32)


# M1: Detect sensitive terms
def is_sensitive_query(query):
    for term in sensitive_terms['term']:
        if term in query:
            return True
    return False


# M2: Match and rank sensitivity level of the query
def check_sensitivity_level(query):
    levels = []
    for index, row in sensitive_terms.iterrows():
        if row['term'] in query:
            levels.append(row['sensitivity_level'])
    return max(levels) if levels else 0


# M3: Evaluate the minimal number of cover queries required
def minimal_cover_queries_required(sensitivity_level):
    return max(1, sensitivity_level)


# M4: Check if required cover queries are available
def cover_queries_available(required_queries):
    return len(cover_queries) >= required_queries


# M6: Generate cover queries using Neural Network
import csv
# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def build_generative_model(input_dim):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_dim,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Reshape((1, 1024)))  # Reshape to add the time step dimension
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(input_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model

# Get BERT embeddings for the cover queries
cover_query_embeddings = np.array([get_bert_embeddings(query) for query in cover_queries['query']])
cover_query_embeddings = cover_query_embeddings.squeeze()  # Remove extra dimensions

# Split the data
X_train, X_test, y_train, y_test = train_test_split(cover_query_embeddings, cover_query_embeddings, test_size=0.2)

# Build the generative model
input_dim = cover_query_embeddings.shape[1]
model = build_generative_model(input_dim)
model.fit(X_train, y_train, epochs=10, batch_size=32)

import csv

import csv


def generate_cover_queries_nn(query, num_queries):
    # Get BERT embeddings for the input query
    query_embedding = get_bert_embeddings(query)

    # Predict with the generative LSTM model
    predicted_embedding = model.predict(query_embedding)

    # Convert the predicted embedding back to a query using the closest vector in the cover queries
    distances = np.linalg.norm(cover_query_embeddings - predicted_embedding, axis=1)
    closest_queries_indices = np.argsort(distances)[:num_queries]
    selected_queries = cover_queries.iloc[closest_queries_indices]['query'].tolist()

    # Save generated queries to CSV
    with open('generated_cover_queries_output.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for generated_query in selected_queries:
            writer.writerow([query, generated_query])

    return selected_queries


# M7: Evaluate if cover queries hide the true intent
def evaluate_cover_queries(cover_queries):
    for cq in cover_queries:
        for term in sensitive_terms['term']:
            if term in cq:
                return False
    return True


# M8: Submit all cover queries with random delay
def submit_queries_with_latency(queries):
    for query in queries:
        delay = random.uniform(0.1, 1.0)
        print(f"Submitting query '{query}' with delay {delay:.2f} seconds")
        time.sleep(delay)


# M9: Show original query results only
def show_original_query_results(query):
    results = query_results[query_results['query'] == query]
    if not results.empty:
        for result in results['result']:
            print(f"Displaying result: {result}")
    else:
        print("No results found for the original query.")


# M10: Apply random click model
def apply_random_click_model(cover_queries):
    print("Applying random click model on cover queries results...")
    clicked_results = []
    for cq in cover_queries:
        results = query_results[query_results['query'].str.contains(cq)]
        if not results.empty:
            clicked_result = random.choice(results['result'].tolist())
            clicked_results.append(clicked_result)
            print(f"Randomly clicked on: {clicked_result}")
        else:
            print(f"No results found for cover query: {cq}")
    return clicked_results


def process_query(query):
    if is_sensitive_query(query):
        sensitivity_level = check_sensitivity_level(query)
        required_cover_queries = minimal_cover_queries_required(sensitivity_level)

        if not cover_queries_available(required_cover_queries):
            cover_queries_list = generate_cover_queries_nn(query, required_cover_queries)
        else:
            cover_queries_list = cover_queries['query'].tolist()[:required_cover_queries]

        if not evaluate_cover_queries(cover_queries_list):
            print("Cover queries are not sufficient to hide the true intent.")
            return

        submit_queries_with_latency(cover_queries_list)
        show_original_query_results(query)
        apply_random_click_model(cover_queries_list)
    else:
        print("Query is not sensitive. Submitting separately.")
        show_original_query_results(query)


# Example usage
query = "sensitive_keyword1 example search"
process_query(query)
