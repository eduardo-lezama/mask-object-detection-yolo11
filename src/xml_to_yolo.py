import os
import xml.etree.ElementTree as ET


# Function to process a single XML file
def process_xml_file(xml_file_path, classes, output_dir):
    """
    Processes an XML annotation file and generates a YOLO-format text file.

    Args:
        xml_file_path (str): Path to the XML file.
        classes (dict): Dictionary mapping class names to their IDs.
        output_dir (str): Directory where the output file will be saved.

    Returns:
        None
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Get image dimensions
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    # Create the output TXT file
    txt_filename = os.path.splitext(os.path.basename(xml_file_path))[0] + ".txt"
    txt_file_path = os.path.join(output_dir, txt_filename)

    with open(txt_file_path, "w") as f:
        for obj in root.iter("object"):
            cls = obj.find("name").text
            cls_id = classes[cls]

            # Get bounding box coordinates
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Calculate normalized coordinates
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # Write to the TXT file
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

# Main function to process all XML files in a directory
def convert_annotations_to_yolo(annotations_dir, classes, output_dir):
    """
    Converts all XML annotation files in a directory to YOLO format (TXT files).

    Args:
        annotations_dir (str): Path to the directory containing XML annotations.
        classes (dict): Dictionary mapping class names to their IDs.
        output_dir (str): Directory where the YOLO TXT files will be saved.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    for xml_file in os.listdir(annotations_dir):
        if xml_file.endswith(".xml"):
            process_xml_file(os.path.join(annotations_dir, xml_file), classes, output_dir)
