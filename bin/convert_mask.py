#!/usr/bin/env python3

import numpy as np
import tifffile
import csv, os
import skimage
import skimage.measure

from argparse import ArgumentParser
from pathlib import Path
from enum import Enum

class TISSUE_FTU(Enum):
    def __new__(
        cls,
        organ_name: str,
        ftu_type: str,
        uberon : str,
        struct : str,    
    ):
        obj = object.__new__(cls)
        obj._value_ = organ_name
        obj.ftu_type = ftu_type
        obj.uberon = uberon
        obj.struct = struct
        return obj

    def __str__(self):
        return self.value

    KIDNEY = ("kidney", "glomeruli", "UBERON:0000074", "UBERON:0001229")
    LUNG = ("lung", "alveoli", "UBERON:0002299", "UBERON:0006524" )
    LARGE_INTESTINE = ("largeintestine", "crypt", "UBERON:0001984", "UBERON:0001207" )
    SPLEEN = ("spleen", "white pulp", "UBERON:0001959", "UBERON:1000023" )
    PROSTATE= ("prostate", "glandular acini", "UBERON:0004179", "UBERON:0000428")

tissue_code_mapping = {
    "LI": TISSUE_FTU.LARGE_INTESTINE,
    "LK": TISSUE_FTU.KIDNEY,
    "RK": TISSUE_FTU.KIDNEY,
    "LL": TISSUE_FTU.LUNG,
    "RL": TISSUE_FTU.LUNG,
    "SP": TISSUE_FTU.SPLEEN,
    "PR": TISSUE_FTU.PROSTATE,
}

def main(ome_tiffs, tissue_code):

    #get mapping info based on organ

    organ = tissue_code_mapping[tissue_code]
    mask_name = organ.ftu_type
    mask_id = organ.uberon
    protocol = "https://github.com/hubmapconsortium/kaggle-2-segmentation/"
    ann_tool = "In-house"
    obj_type = organ.uberon
    an_struct = organ.struct

    with open(f'{mask_name}-objects.tsv', 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        #write the hard-coded header rows
        writer.writerow(["Header Schema ID", "63c06fb2-4638-4979-aa97-5aff2a840156"])
        writer.writerow(["Type", "text", "text", "obi id", "DOI", "category", "obi id", "float", "float", "float", "ontology-id", "float", "float"]) 
        writer.writerow(["Unit", "", "", "", "", "", "", "pixel", "pixel", "pixel", "", "micrometers squared", "micrometers squared"])
        writer.writerow(["Feature class", "Mask", "Mask", "Mask", "Mask", "mask", "ontology", "spatial", "spatial", "spatial", "ontology", "morphology", "morphology"]) 
        writer.writerow(["Description", "The name of the OME TIFF file that contains the mask.", "The name of the mask. THIS MUST MATCH A CHANNEL IN THE OME TIFF FILE.",
                         "The ontological ID that defines the class of objects in the mask. For example, if the mask is of cells, then this would be the ontology ID for \"cell\". In this example, if each object was an identified type of cell, then the \"Object type\" column would capture the ontological ID for the type of cell. This ID must be from an OBO Foundry ontology. https://obofoundry.org/",
                         "The protocol used to define the mask.", "This is the tool that was used to annotate the objects in this mask.",
                         "The ontological ID for the specific object identified in the mask. In some cases the Mask ID and Object type are the same, for example, if the mask is identifying all cells and each object is an unidentified cell. In this case both the mask ID and object type would be \"CL:0000000\".",
                         "The X value of the object’s centroid. The top left corner of the image is assumed to have a pixel value of 0. The OME TIFF file XML header must contain the minimum specification as defined below. Specifically, the physical pixel size is necessary to convert pixel coordinates to spatial coordinates. https://docs.google.com/spreadsheets/d/1YnmdTAA0Z9MKN3OjR3Sca8pz-LNQll91wdQoRPSP6Q4/edit#gid=0",
                         "The Y value of the object’s centroid. The top left corner of the image is assumed to have a pixel value of 0. The OME TIFF file XML header must contain the minimum specification as defined below. Specifically, the physical pixel size is necessary to convert pixel coordinates to spatial coordinates. https://docs.google.com/spreadsheets/d/1YnmdTAA0Z9MKN3OjR3Sca8pz-LNQll91wdQoRPSP6Q4/edit#gid=0",
                         "The Z value of the object’s centroid. The bottom of the 3D object is assumed to have a pixel value of 0. The OME TIFF file XML header must contain the minimum specification as defined below. Specifically, the physical pixel size is necessary to convert pixel coordinates to spatial coordinates. https://docs.google.com/spreadsheets/d/1YnmdTAA0Z9MKN3OjR3Sca8pz-LNQll91wdQoRPSP6Q4/edit#gid=0",
                         "The object is located in an anatomical structure with this ontology identifier.", "This column contains the area value of the object.", "This column contains the radius value of the object."])
        writer.writerow(["Protocol used to derive this feature value (DOI)", "", "", "", "", "", protocol, protocol, protocol, protocol, protocol, protocol, protocol])
        writer.writerow(["Alternative ID"])
        writer.writerow(["Requirement level", "REQUIRED", "REQUIRED", "REQUIRED", "REQUIRED", "REQUIRED", "OPTIONAL", "REQUIRED", "REQUIRED", "REQUIRED", "OPTIONAL", "REQUIRED", "REQUIRED"])
        writer.writerow(["Permissible values", "", "", "", "", "FUSION | Azimuth | STELLAR | In-house | manual | Not applicable"])
        writer.writerow([""])
        writer.writerow(["Ojbect ID", "Source file", "Mask name", "Mask ID", "Protocol for mask creation (DOI)", "Annotation tool", "Object type", "x", "y", "z", "Anatomical structure", "Area", "Radius"]) 

    
        for ome_tiff in ome_tiffs:
            
            with tifffile.TiffFile(ome_tiff) as tif:
                image_data = tif.asarray()
                ome_metadata = tif.ome_metadata

            image_data = skimage.measure.label(image_data)
            image_data = np.array(image_data, dtype=np.uint16) #convert to a type we can actually write
            path_stem = Path(ome_tiff).stem
            tifffile.imwrite(f'{path_stem}.tiff', image_data)
            clusters = skimage.measure.regionprops(image_data)

            filename = os.path.basename(ome_tiff)

            #get and write the data for each object
            for c in clusters:
                centroid_y, centroid_x = c.centroid
                area = c.area
                maj_axis = c.major_axis_length/2
                min_axis = c.minor_axis_length/2
                radius = np.mean([maj_axis, min_axis])
                writer.writerow([c.label, filename, mask_name, mask_id, protocol, ann_tool, obj_type, centroid_x, centroid_y, 0, an_struct, area, radius])


if __name__ == '__main__':

    p = ArgumentParser()
    p.add_argument('ome_tiffs', metavar='PATH', type=str, nargs='+')
    p.add_argument('tissue_code', type=str, choices=sorted(tissue_code_mapping))
    args = p.parse_args()

    main(args.ome_tiffs, args.tissue_code)
