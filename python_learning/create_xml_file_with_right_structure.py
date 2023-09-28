import xml
import xml.etree.ElementTree as ET


class CreateXMLFile:
    def __init__(self):
        self.file_path = "test.xml"


    def create_xml_file(self):
        root = ET.Element("annotations")
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(self.file_path)

    def open(self):
        tree = ET.parse(self.file_path)
        self.root = tree.getroot()
        self.child = ET.SubElement(self.root, "frame", frame="1")

    def add_to_xml_file(self, x, y):
        self.subchild = ET.SubElement(self.child, "ball", points=f"{x},{y}")


    def write_to_file(self):
        tree = ET.ElementTree(self.root)
        tree.write(self.file_path)


def _create_xml_file(file_path: str) -> None:
    root = ET.Element("annotations")
    # tree = ET.ElementTree(root)
    # tree.write(file_path)
    # Add child to root
    child = ET.SubElement(root, "track")
    child.set("id", "0")
    child.set("label", "ball")
    child.set("source", "manual")

    # Add subchild to child
    subchild = ET.SubElement(child, "points")
    subchild.set("frame", "1")
    subchild.set("points", "1,2")

    # Write to file
    tree = ET.ElementTree(root)
    tree.write(file_path)


def add_detections_to_xml(file_path: str, dets) -> None:
    tree = ET.parse(file_path)
    root = tree.getroot()
    print(root.tag)
    print(root.attrib)
    tree.write(file_path)


def main():
    # create_xml_file("test.xml")
    # add_detections_to_xml("test.xml", None)

    xml_file = CreateXMLFile()
    xml_file.create_xml_file()
    xml_file.open()
    for i in range(10):
        xml_file.add_to_xml_file(i, i)
    xml_file.write_to_file()


if __name__ == "__main__":
    main()
