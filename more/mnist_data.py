# import struct
# import numpy as np
# from typing import List
# import os

# def read_mnist_file(filename: str, expected_images: int = -1) -> List[List[float]]:
#     """
#     Βασική συνάρτηση που κάνει ανάγνωση τα MNIST αρχεία
#     """
#     try:
#         with open(filename, 'rb') as file:
#             # Ανάγνωση headers
#             magic = struct.unpack('>I', file.read(4))[0]
#             num_images = struct.unpack('>I', file.read(4))[0]
#             rows = struct.unpack('>I', file.read(4))[0]
#             cols = struct.unpack('>I', file.read(4))[0]
            
#             # Έλεγχος magic number για validity
#             if magic != 2051:
#                 raise ValueError(f"Invalid magic number in MNIST file: {magic}")
            
#             # Έλεγχος συμφωνίας αναμενόμενων και πραγματικών εικόνων
#             if expected_images > 0 and num_images != expected_images:
#                 print(f"Warning: expected {expected_images} images but found {num_images}")
            
#             # Προ-δέσμευση μνήμης για αποδοτικότητα
#             images = []
#             image_size = rows * cols
            
#             for i in range(num_images):
#                 # Ανάγνωση pixel data
#                 buffer = file.read(image_size)
#                 if len(buffer) != image_size:
#                     break
                
#                 # Κανονικοποίηση pixel values στο [0,1]
#                 image = [pixel / 255.0 for pixel in buffer]
#                 images.append(image)
            
#             # print(f"Loaded {len(images)} MNIST images of dimension {image_size}")
#             return images
            
#     except Exception as e:
#         print(f"Error: Cannot open or read file {filename}: {e}")
#         return []

# def return_mnist_data(input_file: str) -> List[List[float]]:
#     """
#     Συνάρτηση για ανάγνωση MNIST training/test data
#     """
#     mnist_data = read_mnist_file(input_file)
#     # print(f"Created MNIST test data: {len(mnist_data)} elements")
#     return mnist_data

# def read_mnist_query(filename: str) -> List[List[float]]:
#     """
#     Συνάρτηση για ανάγνωση MNIST queries
#     """
#     return read_mnist_file(filename)


import struct
import numpy as np
from typing import List
import os

def read_mnist_file(filename: str, expected_images: int = -1, max_images: int = -1) -> List[List[float]]:
    """
    Βασική συνάρτηση που κάνει ανάγνωση τα MNIST αρχεία
    """
    try:
        # Έλεγχος ύπαρξης αρχείου
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")
            
        file_size = os.path.getsize(filename)
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        
        with open(filename, 'rb') as file:
            # Ανάγνωση headers
            magic = struct.unpack('>I', file.read(4))[0]
            num_images = struct.unpack('>I', file.read(4))[0]
            rows = struct.unpack('>I', file.read(4))[0]
            cols = struct.unpack('>I', file.read(4))[0]
            
            print(f"Magic number: {magic}, Images: {num_images}, Dimensions: {rows}x{cols}")
            
            # Έλεγχος magic number για validity
            if magic != 2051:
                raise ValueError(f"Invalid magic number in MNIST file: {magic}")
            
            # Έλεγχος συμφωνίας αναμενόμενων και πραγματικών εικόνων
            if expected_images > 0 and num_images != expected_images:
                print(f"Warning: expected {expected_images} images but found {num_images}")
            
            # Περιορισμός αριθμού εικόνων αν χρειάζεται
            if max_images > 0:
                num_images = min(num_images, max_images)
                print(f"Loading only {num_images} images due to memory constraints")
            
            # Προ-δέσμευση μνήμης για αποδοτικότητα
            images = []
            image_size = rows * cols
            
            for i in range(num_images):
                if i % 10000 == 0 and i > 0:
                    print(f"Processed {i}/{num_images} images...")
                
                # Ανάγνωση pixel data
                buffer = file.read(image_size)
                if len(buffer) != image_size:
                    print(f"Warning: incomplete image data at image {i}")
                    break
                
                # Κανονικοποίηση pixel values στο [0,1]
                image = [pixel / 255.0 for pixel in buffer]
                images.append(image)
            
            print(f"Successfully loaded {len(images)} MNIST images")
            return images
            
    except Exception as e:
        print(f"Error: Cannot open or read file {filename}: {e}")
        return []

def return_mnist_data(input_file: str, max_images: int = -1) -> List[List[float]]:
    """
    Συνάρτηση για ανάγνωση MNIST training/test data
    """
    return read_mnist_file(input_file, max_images=max_images)

def read_mnist_query(filename: str, max_images: int = -1) -> List[List[float]]:
    """
    Συνάρτηση για ανάγνωση MNIST queries
    """
    return read_mnist_file(filename, max_images=max_images)