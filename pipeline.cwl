#!/usr/bin/env cwl-runner
class: Workflow
cwlVersion: v1.1
label: Kaggle-2 segmentation

inputs:
  enable_manhole:
    label: "Whether to enable remote debugging via 'manhole'"
    type: boolean?

  data_directory:
    label: "Path to directory containing raw PAS dataset"
    type: Directory

  tissue_type:
    label: "Code describing the organ from which the sample was derived"
    type: string


outputs:

  ome_tiff_files:
    outputSource: convert/ome_tiff_files
    type: File[]
  json_files:
    outputSource: segmentation/json_files
    type: File[]
  tsv_file:
    outputSource: convert/tsv_file
    type: File

steps:

  - id: segmentation
    in:
      - id: data_directory
        source: data_directory
      - id: tissue_type
        source: tissue_type

    out:
      - ome_tiff_files
      - json_files

    run: steps/segmentation.cwl
    label: "Performs FTU segmentation on H&E data"

  - id: convert
    in:
      - id: ome_tiff_files
        source: segmentation/ome_tiff_files
      - id: tissue_code
        source: tissue_type

    out:
      - ome_tiff_files
      - tsv_file

    run: steps/convert-mask.cwl
    label: "Harmonizes output with EPIC spec"
