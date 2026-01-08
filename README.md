# Hackathon15_AlphaEarth_Alberta
In this repository, the Alpha Earth dataset for Alberta in 2020 is used to evaluate its potential relative to S2 data.
---

1- Run the codes in preprocessig to get dataset and save them in dataset folder

2- Download LULC map using this 
  [link](https://open.canada.ca/data/en/dataset/ee1580ab-a23d-4f86-a09b-79763677eb47/resource/f1ba2faa-ff10-4526-815a-c57b99eef1bb)

3- Save the downloaded LULC map in GroundTruth_Landsat_Canada folder

4- Then run the clip_Landsat8_30m_GT.py to get clipped ground truth LCLU map of Alberta.

---

<table>
  <tr>
    <td align="center">
      <img src="asset/Sentinel2.png" width="250"/><br/>
      <b>Sentinel-2</b>
    </td>
    <td align="center">
      <img src="asset/Landsat8.png" width="250"/><br/>
      <b>Landsat-8</b>
    </td>
    <td align="center">
      <img src="asset/AlphaEarth.png" width="250"/><br/>
      <b>AlphaEarth</b>
    </td>
  </tr>
</table>

---
5- run extract_patches_alldatasets.py to get train, validation, and test patches of Landsat-8, Sentinel-2, and AlphaEarth datasets. It creates train_val_test_patches folder with the following structure:

