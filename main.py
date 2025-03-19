import pandas as pd
from xml.etree import ElementTree as ET
from pathlib import Path

def check_xml_well_formed(xml_path):
    """
    Check if an XML file is well-formed.
    Returns True if well-formed, raises ParseError with details if not.
    """
    try:
        tree = ET.parse(xml_path)
        print(f"✓ XML file '{xml_path}' is well-formed")
        return True
    except ET.ParseError as e:
        print(f"✗ XML parsing error in '{xml_path}':")
        print(f"  Line {e.position[0]}, Column {e.position[1]}")
        print(f"  Error: {str(e)}")
        raise

# Check XML well-formedness before attempting to read
xml_file = 'ICTRP-Results.xml'
if not Path(xml_file).exists():
    print(f"File not found: {xml_file}")
    exit()

check_xml_well_formed(xml_file)
# Read the XML file into a DataFrame using etree parser
df = pd.read_xml(xml_file, parser='etree')

# Column to check if there are multiple values in col countries in given row (values are separated by comma or semicolon)
df['multiple_countries'] = df['Countries'].str.contains(r'[,;]')

# Print number of rows with multiple countries (242 at time of writing)
# print(df['multiple_countries'].sum()) 



# TODO: add unpivoting of what countries are in which row