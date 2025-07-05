Book Recommendation System
This project implements a simple book recommendation system based on textual similarity of book descriptions, reviews, and categories using TF-IDF vectorization and cosine similarity.

Overview
The system:

Loads book and review data from compressed JSON Lines files (.jsonlines.zip).

Merges book metadata and reviews by book_id.

Aggregates reviews per book into a single text feature.

Creates a combined feature consisting of book categories and aggregated reviews.

Builds a TF-IDF vectorizer on these features.

For a given book title input by the user, recommends the top 5 most similar books based on cosine similarity of TF-IDF vectors (excluding the queried book).

Requirements
Python 3.7+

Libraries:

pandas

scikit-learn

tqdm

numpy

You can install the required packages using:

pip install pandas scikit-learn tqdm numpy


File Structure
books.jsonlines.zip: Dataset containing book metadata.

reviews.jsonlines.zip: Dataset containing book reviews.

main.py: Main script implementing the recommendation logic.

The expected relative folder structure (relative to the script location):



Data/
  yakaboo-book-reviews-dataset/
    books.jsonlines.zip
    reviews.jsonlines.zip
Usage
Run the script:


python main.py
You will be prompted to enter a book title. The program will:

Check if the book exists in the dataset.

Compute cosine similarity between the queried book's features and all other books.

Return the top 5 most similar books (excluding the queried one).
