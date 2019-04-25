import json
import os
import random
import shutil
import sys


def main():
    """Process the (pseudo) JSON file and create training/testing directories containing thumbnails.

    $1 -- [str] Path to the (pseudo) JSON file containing data to be fed into MLEP.
    $2 -- [float] Fraction of training data.
    """
    # Create directory structure.
    os.mkdir("training_set")
    os.mkdir("test_set")
    os.mkdir(os.path.join("training_set", "hot"))
    os.mkdir(os.path.join("training_set", "cold"))
    os.mkdir(os.path.join("test_set", "hot"))
    os.mkdir(os.path.join("test_set", "cold"))

    # Load posts data.
    posts = []
    with open(sys.argv[1]) as in_file:
        for in_line in in_file:
            post = json.loads(in_line)
            if post["thumbnail_local_path"]:
                posts.append(post)

    # Separate posts data into 2 sets: training and testing.
    random.shuffle(posts)
    training_posts = posts[:int(len(posts) * float(sys.argv[2]))]
    test_posts = posts[int(len(posts) * float(sys.argv[2])):]

    # Copy training data.
    for post in training_posts:
        shutil.copyfile(
            post["thumbnail_local_path"],
            os.path.join("training_set", "hot" if post["label"] else "cold",
                os.path.basename(post["thumbnail_local_path"]))
        )

    # Copy test data.
    for post in test_posts:
        shutil.copyfile(
            post["thumbnail_local_path"],
            os.path.join("test_set", "hot" if post["label"] else "cold",
                os.path.basename(post["thumbnail_local_path"]))
        )


if __name__ == "__main__":
    main()
