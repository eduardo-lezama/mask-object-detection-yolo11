import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import random

def count_classes_from_txt_files(annotations_dir, classes: dict):
    """
    Counts the number of instances for each class by reading YOLO-format TXT files.

    Args:
        annotations_dir (str): Path to the directory containing YOLO TXT files.
        classes (dict): Dictionary of class names and their IDs.

    Returns:
        dict: A dictionary with class names as keys and their counts as values.
    """
    class_count = {class_name: 0 for class_name in classes.keys()}

    for txt_file in os.listdir(annotations_dir):
        if not txt_file.endswith(".txt"):
            continue
        with open(os.path.join(annotations_dir, txt_file), "r") as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.split()[0])
                class_name = list(class_count.keys())[class_id]
                class_count[class_name] += 1

    return class_count

def plot_class_distribution(class_count: dict):
    """
    Plots the distribution of classes as a bar chart and pie chart

    Args:
        class_count (dict): A dictionary with class names as keys and their counts as values.

    Returns:
        None
    """
    # Plot the bar chart
    plt.bar(class_count.keys(), class_count.values())
    plt.title("Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Number of Instances")
    plt.show()
    
    # Plot the pie chart
    plt.figure(figsize=(4, 4))  # Optional: Adjust the figure size
    plt.pie(class_count.values(), labels=class_count.keys(), autopct='%1.1f%%', startangle=90)
    plt.title("Class Distribution (Proportion)")
    plt.axis('equal')  # Equal aspect ratio ensures the pie is a circle
    plt.show()


def draw_inference_grid(model, dataset_path, grid_size=(3, 3)):
    """
    Plots a grid of inferences with bounding boxes using a YOLO model.

    Args:
        model: YOLO model.
        dataset_path (str): Path to the dataset directory containing images.
        grid_size (tuple): Size of the grid (rows, columns).

    Returns:
        None
    """
    # Obtain images from dataset
    image_paths = [os.path.join(dataset_path, fname) for fname in os.listdir(dataset_path) if fname.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Select random images
    random_images = random.sample(image_paths, grid_size[0] * grid_size[1])

    # Creates Grid
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 10))
    axes = axes.flatten()  
    
    for img_path, ax in zip(random_images, axes):
        # Inference
        results = model.predict(img_path, device="cuda")
        
        # Plot bboxes on orig image
        results_img = results[0].plot()  # Image + bboxes
        results_pil = Image.fromarray(cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB)) 

        # Put image in grid cell
        ax.imshow(results_pil)
        ax.set_title(os.path.basename(img_path))  # Image name as title
        ax.axis("off")  
        
    plt.tight_layout()
    plt.show()
