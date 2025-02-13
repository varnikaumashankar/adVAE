from zipfile import ZipFile 
import gzip
import shutil
import os

def decompress_file(file_path, output_path):
    '''
    Decompressing the input data file (.zip) to the output path.

    Args:
        file_path (str): The path to the input data file.
        output_path (str): The path to the output directory.
    
    Returns:
        None

    '''
    # Check if the file is a .zip file
    if file_path.endswith('.zip'):
        with ZipFile(file_path, 'r') as zip_f:
            zip_f.extractall(output_path)
    
    # Check if the file is a .gz file
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rb') as gz_f:
            with open(output_path, 'wb') as out_f:
                shutil.copyfileobj(gz_f, out_f)

def main():
    # Get the input file and output path from user
    input_file = input('Enter the path to the input data file: ')
    output_path = input('Enter the path to the output directory: ')

    # Check if the input file exists
    if not os.path.exists(input_file):
        print('The input file does not exist.')
        return None
    
    # Check if the output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Decompress the input file if is a .zip file
    if input_file.endswith('.zip'):
        decompress_file(input_file, output_path)
    
if __name__ == '__main__':
    main()
