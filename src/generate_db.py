"""
Use this file to generate the database before using main.py
"""

import random
from typing import List

import numpy as np
from annoy import AnnoyIndex

from default_values import DATABASE_SIZE, VECTOR_DIM

db = AnnoyIndex(VECTOR_DIM, "angular")


def generate_random_vector(dim: int = 64) -> List:
    arr = np.array([random.gauss(0, 1) for z in range(VECTOR_DIM)])
    return [float(x) for x in list(arr / arr.sum())]


for i in range(DATABASE_SIZE):
    v = generate_random_vector(dim=VECTOR_DIM)
    db.add_item(i, v)

db.build(10)  # 10 trees
print("[DONE] Done generating vector database..")

db.save("first_vector_database.ann")
print("[DONE] Saved the database to file:: first_vector_database.ann")
