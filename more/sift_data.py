# sift_data.py

import struct
from typing import List

def read_sift_file(filename: str) -> List[List[float]]:
    """
    Βασική συνάρτηση που κάνει ανάγνωση τα SIFT αρχεία
    """
    try:
        with open(filename, 'rb') as file:
            data = []

            while True:
                # Ανάγνωση διάστασης
                dimension_bytes = file.read(4)
                if not dimension_bytes:
                    break  # End of file
                
                d = struct.unpack('i', dimension_bytes)[0]
                
                # Έλεγχος ότι η διάσταση είναι 128
                if d != 128:
                    print(f"Unexpected dimension: {d}")
                    break
                
                # Ανάγνωση vector
                vector_bytes = file.read(d * 4)  # 4 bytes per float
                if len(vector_bytes) != d * 4:
                    break  # Truncated vector at end-of-file
                
                # Μετατροπή bytes σε list of floats
                vector = list(struct.unpack(f'{d}f', vector_bytes))
                data.append(vector)

                #για testing
                if len(data) >= 1000:
                    break
            
            # print(f"Loaded {len(data)} SIFT vectors")
            return data
            
    except Exception as e:
        print(f"Error: Cannot open file {filename}: {e}")
        return []

def return_sift_data(input_file: str) -> List[List[float]]:
    """
    Συνάρτηση για ανάγνωση SIFT training/test data
    """
    return read_sift_file(input_file)  # "../sift/sift_base.fvecs"

def read_sift_query(filename: str) -> List[List[float]]:
    """
    Συνάρτηση για ανάγνωση SIFT queries
    """
    return read_sift_file(filename)