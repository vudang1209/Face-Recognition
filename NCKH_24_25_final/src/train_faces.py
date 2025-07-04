from database import connect_db, get_all_encodings
import pickle
import numpy as np
import os

def train_encodings():
    conn = connect_db()
    data = get_all_encodings(conn)
    conn.close()

    encoding_dict = {}
    for name, enc in data:
        if name not in encoding_dict:
            encoding_dict[name] = []
        encoding_dict[name].append(np.frombuffer(enc, dtype=np.float32))

    # Tính trung bình cho mỗi người
    final_dict = {
        name: np.mean(enc_list, axis=0) for name, enc_list in encoding_dict.items()
    }

    with open('./src/encodings/encodings.pkl', 'wb') as f:
        pickle.dump(final_dict, f)
    print("Encodings trained and saved to encodings.pkl")

if __name__ == "__main__":
    train_encodings()