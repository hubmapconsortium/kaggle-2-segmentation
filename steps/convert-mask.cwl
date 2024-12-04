cwlVersion: v1.1
class: CommandLineTool
label: converts ometif files to tsv mask

requirements:
  DockerRequirement:
    dockerPull: hubmap/kaggle-2-segmentation:1.1
  DockerGpuRequirement: {}

baseCommand: /opt/convert_mask.py

inputs:
  ome_tiff_files:
    type: File[]
    inputBinding:
      position: 0

  tissue_code:
    type: string
    inputBinding:
      position: 1

outputs:
  tsv_file:
    type: File
    outputBinding:
      glob: "*.tsv"
    doc: tsv file of mask
  ome_tiff_files:
    type: File[]
    outputBinding:
      glob: "*.ome.tiff"
