import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('dir_path')
parser.add_argument('labels_file')
parser.add_argument('new_image_dir')
parser.add_argument('new_labels_file')
args = parser.parse_args()

data = pd.read_csv(os.path.join(args.dir_path, args.labels_file))
data['image'] = '/new_image_dir/' + data['image'].astype(str)

data.to_csv(args.new_labels_file, index=False)
print(data)

