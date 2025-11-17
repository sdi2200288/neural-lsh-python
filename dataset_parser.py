# dataset_parser.py
import struct
import numpy as np
from typing import List


def read_mnist_file(filename: str, expected_images: int = -1) -> List[List[float]]:
    """
    Βασική συνάρτηση που κάνει ανάγνωση τα MNIST αρχεία
    """
    try:
        with open(filename, 'rb') as file:
            # Ανάγνωση headers
            magic = struct.unpack('>I', file.read(4))[0]
            num_images = struct.unpack('>I', file.read(4))[0]
            rows = struct.unpack('>I', file.read(4))[0]
            cols = struct.unpack('>I', file.read(4))[0]
            
            # Έλεγχος magic number για validity
            if magic != 2051:
                raise ValueError(f"Invalid magic number in MNIST file: {magic}")
            
            # Έλεγχος συμφωνίας αναμενόμενων και πραγματικών εικόνων
            if expected_images > 0 and num_images != expected_images:
                print(f"Warning: expected {expected_images} images but found {num_images}")
            
            # Προ-δέσμευση μνήμης για αποδοτικότητα
            images = []
            image_size = rows * cols
            
            for i in range(num_images):
                # Ανάγνωση pixel data
                buffer = file.read(image_size)
                if len(buffer) != image_size:
                    break
                
                # Κανονικοποίηση pixel values στο [0,1]
                image = [pixel / 255.0 for pixel in buffer]
                images.append(image)
            
            # print(f"Loaded {len(images)} MNIST images of dimension {image_size}")
            return images
            
    except Exception as e:
        print(f"Error: Cannot open or read file {filename}: {e}")
        return []

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

def return_data(input_file: str, data_type: str) -> List[List[float]]:
    """
    Συνάρτηση για ανάγνωση MNIST training/test data
    """
   
    if(data_type == "sift"):
        data = read_sift_file(input_file);
    elif(data_type == "mnist"):
        data = read_mnist_file(input_file);
    else:
        raise ValueError(f"Unknown data type: {data_type}. Use 'sift' or 'mnist'")
    
    return data


# Test functions
if __name__ == "__main__":
    # Test with a small file first
    try:
        # You'll need to replace with actual file paths
        mnist_data = return_data("mnist_data/train-images-idx3-ubyte", "mnist")
        sift_data = return_data("sift_data/sift_base.fvecs", "sift")
        print("Parser ready!")
        print(f"MNIST data: {len(mnist_data)} images")
        print(f"SIFT data: {len(sift_data)} vectors")
    except Exception as e:
        print(f"Error: {e}")


