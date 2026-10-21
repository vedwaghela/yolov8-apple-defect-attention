# Attribution and reuse

## SSDA dataset

B. N. H. Nguyen, Surface Subtle Defects Apple dataset, version 1.0.0 (2024).
Dataset: https://doi.org/10.5281/zenodo.10957715
Source: https://github.com/nbngochan/Surface-Subtle-Defects-Apple
Revision: 71fc44fa3358e5cbc0a59d574d38762aaefd676b
Recorded dataset license: Creative Commons Attribution 4.0 International.
Terms: https://creativecommons.org/licenses/by/4.0/

Full dataset images and labels are not redistributed here. Prediction previews and figures contain dataset-derived imagery, credited to the dataset creator under the same attribution terms. Preparation clips one out-of-bounds annotation; the study uses new grouped partitions. Class names are inferred from count agreement with the HAFREE preprint, not an authoritative recovered class dictionary.

Related benchmark: HAFREE, https://doi.org/10.1109/ACCESS.2024.3504399

## Attention mechanisms

- SE: Hu, Shen and Sun, https://doi.org/10.1109/CVPR.2018.00745
- CBAM: Woo et al., https://doi.org/10.1007/978-3-030-01234-2_1
- Coordinate Attention: Hou, Zhou and Feng, CVPR 2021; https://openaccess.thecvf.com/content/CVPR2021/html/Hou_Coordinate_Attention_for_Efficient_Mobile_Network_Design_CVPR_2021_paper.html

Implementation details and source links appear in baseline/ATTENTION.md. Consult the original implementations' terms before redistributing adaptations.

## Software and weights

The experiment uses Ultralytics 8.3.203, PyTorch and their dependencies. Third-party software and pretrained weights retain their original licenses; this repository does not override them. Full third-party source trees and model weights are not bundled. Review applicable dependency and model terms before redistribution or commercial use.

No blanket license for the authors' code, report or slides has been selected in this package. Public availability alone is not a grant of unrestricted reuse rights.
