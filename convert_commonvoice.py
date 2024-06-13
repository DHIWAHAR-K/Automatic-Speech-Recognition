#convert_commonvoice.py
import os
import argparse
import json
import random
import csv
from pydub import AudioSegment

def main(args):
    data = []  # List to store data entries
    directory = os.path.dirname(args.file_path)  # Get directory from the file path
    percent = args.percent  # Percentage of data for testing

    # Calculate total number of lines in the TSV file
    with open(args.file_path) as f:
        length = sum(1 for line in f)

    # Read the TSV file and process each row
    with open(args.file_path, newline='') as csvfile: 
        reader = csv.DictReader(csvfile, delimiter='\t')
        index = 1
        if args.convert:
            print(f"{length} files found")
        for row in reader:  
            file_name = row['path']
            filename = os.path.splitext(file_name)[0] + ".wav"
            text = row['sentence']

            if args.convert:
                data.append({
                    "key": os.path.join(directory, "clips", filename),
                    "text": text
                })
                print(f"Converting file {index}/{length} to wav", end="\r")
                src = os.path.join(directory, "clips", file_name)
                dst = os.path.join(directory, "clips", filename)
                sound = AudioSegment.from_mp3(src)
                sound.export(dst, format="wav")
                index += 1
            else:
                data.append({
                    "key": os.path.join(directory, "clips", file_name),
                    "text": text
                })
    
    random.shuffle(data)  # Shuffle the data

    # Create training JSON file
    print("Creating JSON files")
    train_path = os.path.join(args.save_json_path, "train.json")
    with open(train_path, 'w') as f:
        d = len(data)
        for i in range(int(d - d / percent)):
            line = json.dumps(data[i])
            f.write(line + "\n")
    
    # Create testing JSON file
    test_path = os.path.join(args.save_json_path, "test.json")
    with open(test_path, 'w') as f:
        d = len(data)
        for i in range(int(d - d / percent), d):
            line = json.dumps(data[i])
            f.write(line + "\n")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Utility script to convert commonvoice into wav and create the training and test json files for speech recognition.
    """)
    parser.add_argument('--file_path', type=str, required=True,
                        help='Path to one of the .tsv files found in cv-corpus')
    parser.add_argument('--save_json_path', type=str, required=True,
                        help='Path to the directory where the JSON files are supposed to be saved')
    parser.add_argument('--percent', type=int, default=10, required=False,
                        help='Percent of clips to be put into test.json instead of train.json')
    parser.add_argument('--convert', default=True, action='store_true',
                        help='Convert MP3 files to WAV')
    parser.add_argument('--not-convert', dest='convert', action='store_false',
                        help='Do not convert MP3 files to WAV')
    
    args = parser.parse_args()
    main(args)