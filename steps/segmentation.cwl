cwlVersion: v1.1
class: CommandLineTool
label: segments each image in the directory for FTUs

requirements:
  DockerRequirement:
    dockerPull: hubmap/kaggle-2-segmentation:1.0.1
  DockerGpuRequirement: {}

baseCommand: /opt/inference.py

inputs:
  enable_manhole:
    label: "Whether to enable remote debugging via 'manhole'"
    type: boolean?
    inputBinding:
      position: 0

  data_directory:
    type: Directory
    doc: Path to processed dataset directory
    inputBinding:
      position: 1

  tissue_type:
    label: "Name of organ that sample is derived from"
    type: string
    inputBinding:
      position: 2


outputs:
  ome_tiff_files:
    type: File[]
    outputBinding:
      glob: "*.ome.tiff"
    doc: binary segmentation masks in ome tiff form

  json_files:
    type: File[]
    outputBinding:
      glob: "*.json"
    doc: indexed segmentation masks in geoJSON format
