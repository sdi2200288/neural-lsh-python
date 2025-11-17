# dataset_parser.py
import struct
import numpy as np

def read_mnist_images(filename):
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

def read_sift_vectors(filename):
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

# Test functions
if __name__ == "__main__":
    # Test with a small file first
    print("Testing MNIST parser...")
    try:
        # ΠΡΟΣΟΧΗ: Άλλαξε τα paths αν τα αρχεία σου είναι αλλού
        mnist_data = read_mnist_images("mnist_data/train-images-idx3-ubyte")
        print(f"MNIST data shape: {mnist_data.shape}")
    except Exception as e:
        print(f"MNIST error: {e}")
    
    print("\nTesting SIFT parser...")
    try:
        sift_data = read_sift_vectors("sift/sift_base.fvecs")
        print(f"SIFT data shape: {sift_data.shape}")
    except Exception as e:
        print(f"SIFT error: {e}")