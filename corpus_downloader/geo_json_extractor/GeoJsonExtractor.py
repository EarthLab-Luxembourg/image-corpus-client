import json
import os
import logging
import re

LOGGER = logging.getLogger(__name__)

class GeoJsonExtractor(object):
    def __init__(self):
        self.empty_geo_json=''

    @classmethod
    def getGeoJson(path: str, image_name : str, labels : list):
        label_name = image_name.split('.')[0]
        file_name  = find_label_file(label_name)
        geo_json = extract_labels_and_polygons(file_name=file_name, path=path)
        return geo_json

    def find_label_file(label_name : str, labels : list):
        for label in labels:
            if label_name in label:
                return label
            else:
                LOGGER.error("No label found for "+label_name+" image")
                return 'No labels'

    def extract_labels_and_polygons(file_name : str, path : str):
        """This function need to be customize accordingly to the dataset"""

        features = []

        f = open(path+'/'+file_name, "rb")
        try:
            lines = f.readlines()
        finally:
            f.close()
        lines = lines[2:]

        for line in lines:
            decode_line = line.decode("utf-8")
            digits = re.findall('\d+', decode_line)
            label_array = re.findall('[a-zA-Z]+', decode_line)
            """Convert text data to GeoJson data"""
            if(len(label_array) < 2):
                label = label_array[0]
            else:
                label = label_array[0]+' '+label_array[1]
            coordinates = [[float(digits[0]), float(digits[1])], [float(digits[2]), float(digits[3])], [float(digits[4]), float(digits[5])], [float(digits[6]), float(digits[7])]]

            feature =  {"type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": coordinates
             },
             "properties": {
               "label": label,
               }
            }
            features.append(feature)

        geo_json = { "type": "FeatureCollection",
                    "features": features}
        return geo_json
