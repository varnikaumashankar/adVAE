from zipfile import ZipFile 
import gzip
import shutil
import os

def decompress_file(file_path, output_path):
    """
    Decompressing the input data file (.zip) to the output path.

    Args:
        file_path (str): The path to the input data file.
        output_path (str): The path to the output directory.
    
    Returns:
        None
    """
    # Check if the file is a .zip file
    if file_path.endswith('.zip'):
        with ZipFile(file_path, 'r') as zip_f:
            zip_f.extractall(output_path)
        return output_path  # Return the directory where files were extracted
    
    # Check if the file is a .gz file
    if file_path.endswith('.gz'):
        output_file_path = os.path.join(output_path, os.path.basename(file_path).replace('.gz', ''))
        with gzip.open(file_path, 'rb') as gz_f:
            with open(output_file_path, 'wb') as out_f:
                shutil.copyfileobj(gz_f, out_f)
        return output_file_path  # Return the path to the decompressed file

    return None

