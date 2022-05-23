import click
from pathlib import Path
import os
import glob
import hashlib
from shutil import copyfile, copytree
from typing import Dict, Tuple, List
from utils.time_utils import timefunc


@click.command()
@click.option(
    "--user_data",
    default="data/andrew-ng-dcai-comp-2021-data-original/andrew-ng-dcai-comp-2021-data/",
    help="Path to the training and validation data",
)
@timefunc
def main(user_data):

    # copy data
    src_data = user_data
    dst_data = (
        "data/andrew-ng-dcai-comp-2021-data-deduped/andrew-ng-dcai-comp-2021-data/"
    )
    print(f"Copying data from {src_data}...")
    print(f"...to: {dst_data}")
    copytree(src_data, dst_data)

    # find the dupes
    file_hash, dupes_to_remove, files_to_keep = find_dupes_in_data_dir(dst_data)

    # remove the dupes
    print(f"Removing {len(dupes_to_remove)} dupes...")
    for file_path in dupes_to_remove:
        print(f"  {file_path}")
        os.remove(file_path)


def find_dupes_in_data_split_dir(
    data_dir: str, file_hash: Dict = {}, dupes_to_remove: List = [], verbose=False
) -> Tuple[Dict, List]:

    # data_dir corresponds to the train, val, or test folder
    # use hash table to detect duplicate images (same image but different file names)

    for file_name in glob.iglob(data_dir + "/**", recursive=True):
        if os.path.isfile(file_name):

            with open(file_name, "rb") as f:
                hash = hashlib.sha256(f.read()).hexdigest()

            if hash in file_hash.keys():
                if verbose:
                    print("----")
                    print("Duplicate found!")
                    print(f"  Image already in hash table: {file_hash[hash]}")
                    print(f"  Duplicate image: {file_name}")
                dupes_to_remove.append(file_name)
            else:
                file_hash[hash] = file_name

    return file_hash, dupes_to_remove


def find_dupes_in_data_dir(user_data):

    # user_data contains train and val folders

    train_path = user_data + "/train"
    val_path = user_data + "/val"

    print(f"Finding dupes in {train_path}")
    file_hash, dupes_to_remove = find_dupes_in_data_split_dir(train_path)

    print(f"Finding dupes in {val_path}")
    file_hash, dupes_to_remove = find_dupes_in_data_split_dir(
        val_path, file_hash, dupes_to_remove
    )

    # files to keep from hash table
    files_to_keep = list(file_hash.values())

    print(f"Number of dupes to remove: {len(dupes_to_remove)}")
    print(f"Number of files to keep: {len(file_hash.keys())}")

    return file_hash, dupes_to_remove, files_to_keep


if __name__ == "__main__":
    main()
