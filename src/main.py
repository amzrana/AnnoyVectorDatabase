"""
Use this file to extract top K items in database.
This file assumes that you have generated the database using 
generate_db.py file.
"""

import argparse
import random

import pandas as pd
from annoy import AnnoyIndex

from default_values import VECTOR_DIM


def load_database():
    db = AnnoyIndex(VECTOR_DIM, "angular")
    db.load("first_vector_database.ann")
    return db


def print_top_k_indices(query_response, top_k_count: int):
    print(f"Top {top_k_count} indices are:")
    df_dict = {
        "indices": query_response[0],
        "distances": query_response[1],
    }
    df = pd.DataFrame(df_dict)
    print(df)


def print_top_k_vectors(query_response, database):
    top_k_vectors = [database.get_item_vector(idx) for idx in query_response]
    print(top_k_vectors)


def main(top_k_count: int):
    db = load_database()

    # Extract the top K indices
    top_k_indices_with_distance = db.get_nns_by_vector(
        [random.gauss(0, 1) for _ in range(VECTOR_DIM)],
        top_k_count,
        search_k=-1,
        include_distances=True,
    )

    # print top K indices
    print_top_k_indices(top_k_indices_with_distance, top_k_count)

    # # print the top K vectors
    # print_top_k_vectors(top_k_indices_with_distance, db)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", "-k", default=3, type=int)
    args = parser.parse_args()

    main(args.top_k)
