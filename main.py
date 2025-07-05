import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import json
import numpy as np
from tqdm import tqdm
import os


def load_zip_to_df(zip_file_path: str) -> pd.DataFrame:

    data = []

    with zipfile.ZipFile(zip_file_path) as zip_ref:
        file_name = zip_ref.namelist()[0]
        with zip_ref.open(file_name) as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Loading {file_name}"):
            decoded_line = line.decode("utf-8", errors="replace")
            try:
                data.append(json.loads(decoded_line))
            except json.JSONDecodeError as e:
                print("Помилка JSON:", e)
                continue

    df = pd.DataFrame(data)

    return df


def merge_two_tables(first_df, second_df) -> pd.DataFrame:

    df = first_df.merge(second_df, on='book_id')

    return df


def create_features_id_df(df_merge) -> pd.DataFrame:

    df_merge = df_merge.groupby("title")[
        ["titleEnglish", "description", "attrs", "rating_x", "review", "book_id", "categories"]].agg(
        {
            "titleEnglish": "first",
            "description": "first",
            "attrs": "first",
            "rating_x": "first",
            "review": lambda x: " ".join(x),
            "book_id": "first",
            "categories": "first",
        }
    ).reset_index()
    df_merge = df_merge.sort_values("book_id", ascending=True)

    df_merge["features"] = df_merge["categories"].apply(lambda x: " ".join(x)) + df_merge["review"]

    return df_merge


def create_and_fit_vectorizer(df) -> TfidfVectorizer:

    vectorizer = TfidfVectorizer()
    vectorizer.fit(df["features"])

    return vectorizer


def get_recommendation(query_str: str, df: pd.DataFrame):
    vectorizer = create_and_fit_vectorizer(df)

    matched_row = df[df["title"].str.lower() == query_str.lower()]
    if matched_row.empty:
        print(f"Книгу '{query_str}' не знайдено.")
        return pd.DataFrame()

    query_feature = matched_row["features"].values[0]
    query_vect = vectorizer.transform([query_feature])

    word_matrix = vectorizer.transform(df["features"])

    similarities = cosine_similarity(query_vect, word_matrix).flatten()

    query_idx = matched_row.index[0]
    similarities[query_idx] = -np.inf

    top5_indices = np.argpartition(-similarities, 5)[1:6]
    top5_indices = top5_indices[np.argsort(-similarities[top5_indices])]

    print("Найбільш схожі індекси:", top5_indices)
    print("Схожості:", similarities[top5_indices])

    return df.iloc[top5_indices]


def main():
    input_title = input("Введіть назву книги: ")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    books_path = os.path.join(BASE_DIR, "Data", "yakaboo-book-reviews-dataset", "books.jsonlines.zip")
    reviews_path = os.path.join(BASE_DIR, "Data", "yakaboo-book-reviews-dataset", "reviews.jsonlines.zip")

    books = load_zip_to_df(books_path)
    reviews = load_zip_to_df(reviews_path)

    df = merge_two_tables(books, reviews)
    df = create_features_id_df(df)

    result = get_recommendation(input_title, df)
    print(result.columns)
    print(result)

    return result


if __name__ == "__main__":
    main()
