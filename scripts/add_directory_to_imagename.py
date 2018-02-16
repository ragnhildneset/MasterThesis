import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('dir_path')
parser.add_argument('new_image_dir')
args = parser.parse_args()

data = pd.read_csv(os.path.join(args.dir_path, "labels.csv"))
data['image'] = args.new_image_dir + "/" + data['image'].astype(str)

data.to_csv(args.new_image_dir + ".csv", index=False)
print(data)

