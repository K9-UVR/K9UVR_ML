from absl import app, flags, logging
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

FLAGS = flags.FLAGS
flags.DEFINE_string("path_to_imgs", os.getcwd(), "Path to image files.")
flags.DEFINE_string("output_name", "converted_labels.csv",
                    "Name of converted filename.")

# Required flag.
# flags.mark_flag_as_required("output_name")


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member.find('name').text,
                     int(bndbox.find('xmin').text),
                     int(bndbox.find('ymin').text),
                     int(bndbox.find('xmax').text),
                     int(bndbox.find('ymax').text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main(argv):
    del argv  # Unused.
    image_path = os.path.join(FLAGS.path_to_imgs, 'annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(FLAGS.output_name, index=None)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    app.run(main)
