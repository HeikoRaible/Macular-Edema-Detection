# Macular Edema Detection

The aim of this project was to use deep learning methods on a system at the Uniklinik Münster to detect macular edemas in OCT scans carried out there, measure the size of them via counting segmented pixels, and automatically integrate this information into the Fidus software used at the Uniklinik Münster. The basic idea for implementing the deep learning methods was a two-stage process: using an Efficient-Net as a pre-filter, followed by a Mask R-CNN as a segmentation model. Both models individually provided satisfactory results. However, since the segmentation model alone led to better results than the combination of both models in a two-stage process, only this model was integrated into the final prototype. While the quality improvement of the predictions by the pre-filter classification was questioned, it appears promising and requires further investigation. With the Mask R-CNN segmentation model and a script that integrates the results of the model into the Fidus software, the set goals were satisfactorily achieved, proving its technical applicability in practice.

## Labeling

34.943 OCT Scans have been manually labeled using VGG Image Annotator to generate training and test data. There are 25 cross-section images per eye.

## Model

The segmentation model used was a Mask-R CNN.

## Documentation

More information can be found in the documentation at `documentation.pdf`.

## Examples

![IoU=0.41](documentation\pic\Segmentierung\Segmentierungsergebnisse\7.PNG)
![IoU=0.43](documentation\pic\Segmentierung\Segmentierungsergebnisse\9.PNG)
![IoU=0.69](documentation\pic\Segmentierung\Segmentierungsergebnisse\14.PNG)
![IoU=0.54](documentation\pic\Segmentierung\Segmentierungsergebnisse\16.PNG)
![IoU=0.87](documentation\pic\Segmentierung\Segmentierungsergebnisse\40.PNG)
![IoU=0.88](documentation\pic\Segmentierung\Segmentierungsergebnisse\44.PNG)