import os
import shutil
import random
from pathlib import Path

def identify_minority_images(annotations_dir, minority_classes):
    """
    Identifies images containing minority classes based on YOLO TXT annotations.

    Args:
        annotations_dir (str): Path to the directory containing TXT annotation files.
        minority_classes (list): List of minority class IDs.

    Returns:
        dict: Dictionary with class IDs as keys and lists of corresponding image filenames as values.
    """
    minority_images = {cls_id: [] for cls_id in minority_classes}

    for txt_file in os.listdir(annotations_dir):
        if txt_file.endswith(".txt"):
            with open(os.path.join(annotations_dir, txt_file), "r") as f:
                classes_in_image = {int(line.split()[0]) for line in f.readlines()}
                for cls_id in minority_classes:
                    if cls_id in classes_in_image:
                        minority_images[cls_id].append(txt_file.replace(".txt", ".png"))

    return minority_images

def split_images(image_list, train_ratio, val_ratio):
    """
    Splits a list of images into train and validation sets.

    Args:
        image_list (list): List of image filenames.
        train_ratio (float): Proportion of images for training.
        val_ratio (float): Proportion of images for validation.

    Returns:
        tuple: Two lists (train_set, val_set).
    """
    random.shuffle(image_list)
    train_split = image_list[:int(train_ratio * len(image_list))]
    val_split = image_list[int(train_ratio * len(image_list)):int((train_ratio + val_ratio) * len(image_list))]
    return train_split, val_split

def get_repeated_images(minority_images):
    """
    Identifies images that are present in multiple minority class lists.

    Args:
        minority_images (dict): Dictionary with class IDs as keys and lists of corresponding image filenames as values.

    Returns:
        list: List of image filenames that are present in multiple minority class lists.
    """
    # Convert each class's image list to a set
    sets_of_images = [set(images) for images in minority_images.values()]

    # Find the intersection of all sets to get repeated images
    repeated_images = set.intersection(*sets_of_images) if sets_of_images else set()

    return list(repeated_images)

def handle_data_leakage(minority_images, repeated_images, train_ratio, val_ratio):
    """
    Removes leaked images (present in multiple minority class lists) and splits the repeated images.

    Args:
        minority_images (dict): Dictionary of minority class images.
        repeated_images (list): List of images present in multiple minority lists.
        train_ratio (float): Proportion of images for the training split.
        val_ratio (float): Proportion of images for the validation split.

    Returns:
        dict: Updated minority images dictionary without leaked images.
        tuple: Train and validation sets for repeated images.
    """
    for cls_id in minority_images.keys():
        minority_images[cls_id] = list(set(minority_images[cls_id]) - set(repeated_images))

    random.shuffle(repeated_images)
    train_shared = repeated_images[:int(train_ratio * len(repeated_images))]
    val_shared = repeated_images[int(train_ratio * len(repeated_images)):int((train_ratio + val_ratio) * len(repeated_images))]

    return minority_images, train_shared, val_shared

def copy_files(file_list, dest_dir, src_img_dir, src_annot_dir):
    """
    Copies images and annotations to a specific directory.

    Args:
        file_list (list): List of filenames to copy.
        dest_dir (str): Destination directory.
        src_img_dir (str): Source directory for images.
        src_annot_dir (str): Source directory for annotations.

    Returns:
        None
    """
    for file in file_list:
        image_path = os.path.join(src_img_dir, file)
        annot_path = os.path.join(src_annot_dir, file.replace(".png", ".txt"))
        try:
            shutil.copy(image_path, os.path.join(dest_dir, file))
            shutil.copy(annot_path, os.path.join(dest_dir, file.replace(".png", ".txt")))
        except FileNotFoundError:
            print(f"File not found: {file}")