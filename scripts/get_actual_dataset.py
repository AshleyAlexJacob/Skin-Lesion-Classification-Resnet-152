import os

DATA_DIR = "data/processed"

if __name__ == "__main__":
    total_dataset_size = 0
    for category in os.listdir(DATA_DIR):

        if category == ".DS_Store" or category == ".keep":
            continue
        print(category)
        category_size = len(os.listdir(os.path.join(DATA_DIR, category)))
        print(category_size)
        total_dataset_size += category_size
    print(f"Total dataset size: {total_dataset_size}")