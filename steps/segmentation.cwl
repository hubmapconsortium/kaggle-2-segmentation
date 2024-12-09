cwlVersion: v1.1
class: CommandLineTool
label: segments each image in the directory for FTUs

requirements:
  DockerRequirement:
    dockerPull: hubmap/kaggle-2-segmentation:1.1.1
  DockerGpuRequirement: {}
  EnvVarRequirement:
    envDef:
      CUDA_VISIBLE_DEVICES: "0"


baseCommand: /opt/inference.py

inputs:
  data_directory:
    type: Directory
    doc: Path to processed dataset directory
    inputBinding:
      position: 0
      prefix: "--data_directory"

  tissue_type:
    label: "Code of organ that sample is derived from, e.g. RL, SP"
    type: string
    inputBinding:
      position: 1
      prefix: "--tissue_code"


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
