# dataset_parser.py
import struct
import numpy as np
from typing import List

def  read_mnist_file(filename: str, expected_images: int = -1) -> List[List[float]]:
    """Διαβάζει MNIST αρχεία - Βάση από τον C++ κώδικά σου"""
    with open(filename, 'rb') as f:
        # Read header
        magic = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        
        print(f"Reading MNIST: {num_images} images, {rows}x{cols}")
        
        # Read image data
        images = []
        for i in range(num_images):
            img_data = np.frombuffer(f.read(rows * cols), dtype=np.uint8)
            images.append(img_data.astype(np.float32) / 255.0)  # Normalize to [0,1]
        
        return np.array(images)

def read_sift_file(filename: str) -> List[List[float]]:
    """Διαβάζει SIFT αρχεία - Βάση από τον C++ κώδικά σου"""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            # Read dimension
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
                
            dim = struct.unpack('I', dim_bytes)[0]
            if dim != 128:
                print(f"Unexpected dimension: {dim}")
                break
                
            # Read vector
            vec_bytes = f.read(128 * 4)  # 128 floats * 4 bytes each
            if len(vec_bytes) != 128 * 4:
                break
                
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            vectors.append(vec)
    
    print(f"Loaded {len(vectors)} SIFT vectors")
    return np.array(vectors)


def return_data(input_file: str, data_type: str) -> List[List[float]]:
    """
    Συνάρτηση για ανάγνωση MNIST training/test data
    """
   
    if(data_type == "sift"):
        data = read_sift_file(input_file)
    elif(data_type == "mnist"):
        data = read_mnist_file(input_file)
    else:
        raise ValueError(f"Unknown data type: {data_type}. Use 'sift' or 'mnist'")
    
    return data


# Test functions
if __name__ == "__main__":
    # Test with a small file first
    try:
        # You'll need to replace with actual file paths
        mnist_data = return_data("mnist_data/train-images-idx3-ubyte", "mnist")
        sift_data = return_data("sift/sift_base.fvecs", "sift")
        print("Parser ready!")
        # print(f"MNIST data shape: {mnist_data.shape}")
        print(f"MNIST data: {len(mnist_data)} images")
        # print(f"SIFT data shape: {sift_data.shape}")
        print(f"SIFT data: {len(sift_data)} vectors")
        
    except Exception as e:
        print(f"Error: {e}")


