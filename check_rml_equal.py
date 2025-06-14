import xml.etree.ElementTree as ET

def xml_elements_equal(e1, e2):
    if e1.tag != e2.tag or e1.text != e2.text or e1.attrib != e2.attrib:
        return False
    if len(e1) != len(e2):
        return False
    return all(xml_elements_equal(c1, c2) for c1, c2 in zip(e1, e2))

def is_rml_structure_equal(file1, file2):
    tree1 = ET.parse(file1)
    tree2 = ET.parse(file2)
    root1 = tree1.getroot()
    root2 = tree2.getroot()
    return xml_elements_equal(root1, root2)

if __name__ == "__main__":
    print(is_rml_structure_equal('1.rml', '2.rml'))  # True or False
