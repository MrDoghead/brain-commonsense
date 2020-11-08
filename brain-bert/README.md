## Session1

In this session, we aim to fine-tune bert model on brain data (FMRI) in order to find the coorelation between voxels and bert layers.

The FMRI and voxel data can be downloaded [here](https://drive.google.com/drive/folders/1Q6zVCAJtKuLOh-zWpkS3lH8LBvHcEOE8).

## To extract bert features

```bash
sh run_extract.sh
```

## To find highly related voxels

```bash
sh run_predict.sh
```
