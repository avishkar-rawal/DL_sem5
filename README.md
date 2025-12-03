# YOLOv11-Based Segmentation for Tree Canopy Mapping
## 1. Introduction

Tree canopy segmentation refers to the automated identification and delineation of individual or clustered tree crowns from high-resolution aerial or satellite imagery. Although widely applied across environmental and geospatial domains, existing segmentation methods often struggle to generalize across regions, imaging conditions, and mixed urban–rural landscapes. Variations in lighting, canopy density, background textures, and structural noise make consistent model performance challenging, emphasizing the need for more robust and adaptable approaches. To encourage progress in this area, Solafune introduced a dedicated tree-canopy-detection competition that brings together diverse aerial and satellite imagery and provides a rigorous benchmark for evaluating modern segmentation models. As part of this challenge, we reviewed a range of classical and contemporary architectures and ultimately selected YOLOv11, the current state of the art in real-time segmentation, for its strong performance and generalization capabilities.

## 2. Dataset description

The competition dataset is constructed from a curated collection of imagery sourced from SWISSIMAGE, NIAP, and national New Zealand aerial datasets. It includes high-resolution RGB TIFF images spanning both rural and urban environments, with spatial resolutions ranging from 10 cm to 80 cm. Polygon-annotated tree canopy masks are provided for the training split, accompanied by 150 evaluation images used for model validation. The dataset reflects real-world complexity, containing varied canopy structures, shadows, rooftops, mixed vegetation, roads, and other forms of environmental noise, making it a representative foundation for developing tree canopy segmentation models.


The polygon-based canopy annotations provided in the dataset use standard evaluation metric mean Average Precision (mAP-75) score.

<img width="1612" height="544" alt="image" src="https://github.com/user-attachments/assets/4ab2e385-417e-43a0-ae72-44a575d9e914" />

## 3. Solutions Proposed by Others
   
Research in Individual Tree Crown Detection (ITCD) has evolved substantially over the past decade, moving from traditional image-processing techniques toward deep learning–based architectures capable of modeling complex canopy structures. A comprehensive review of the field shows that, beginning around 2017, deep learning (DL) rapidly became the dominant methodological choice.  Within semantic segmentation, several architectures—such as U-Net, DeepLabV3+, SegNet, and FCN have been widely adopted for wall-to-wall canopy delineation. Among these, U-Net consistently emerges as a strong baseline, achieving F1-scores between 94.00% and 94.31% in comparative studies and performing competitively with contemporary models like DeepLabV3+. Its encoder–decoder structure effectively captures broad spatial context and canopy shape, enabling accurate mapping of continuous vegetative cover. However, a well-documented limitation is its tendency to merge overlapping crowns into single segmentation masks. Because it performs pixel-wise classification without explicit instance separation, performance decreases in dense urban environments where tree crowns are tightly clustered and structurally complex.

Deep learning’s superiority becomes even more apparent when contrasted with traditional machine learning (TML) methods such as Support Vector Machines (SVM) and Random Forests (RF). Studies mapping Urban Tree Canopy (UTC) coverage using NAIP imagery reported that U-Net achieved 91.4% overall accuracy in Laurel, MS, and 89.8% in Georgetown, TX substantially higher than SVM (84.2% and 68.6%) and RF (76.2% and 71.3%) in the same settings. While TML methods offer faster training times, they struggle to capture fine-scale canopy details and irregular crown boundaries, leading to noticeably poorer performance in heterogeneous landscapes.
To address challenges in dense or cluttered canopies, instance segmentation models have gained traction. BlendMask, in particular, has shown strong ability to separate adjacent crowns by modeling tree instances individually rather than merging them. In a multi-species forest inventory, BlendMask achieved 92.68% overall accuracy and an F1-score of 81.84%, surpassing U-Net’s 86.04% accuracy and 74.10% F1-score. Its architectural design, especially the top module responsible for high-resolution feature refinement, enables better extraction of crown contours and improves delineation in highly crowded scenes.

Parallel to these developments, YOLO-based instance segmentation models have been explored for real-time or large-area ITCD applications. A recent study applied YOLOv7 to 0.5 m very high-resolution satellite imagery over Ahmedabad, tiling the raster into more than 6,500 patches to enable tree-level and canopy-level change detection. YOLOv7 achieved a mAP of 0.715 for individual tree detection and 0.699 for canopy mask extraction, with tuned configurations reaching up to 80% detection accuracy and a 2% false segmentation rate. Despite its strong cluster separation and speed, YOLOv7 was limited primarily by data resolution: individual crown delineation typically requires imagery between 5 cm and 15 cm, whereas the available 0.5 m data caused significant performance drops in dense urban regions due to pixel insufficiency, shadows, and structural noise.

Recent reviews emphasize that no single architecture robustly handles all canopy scenarios. As a result, ensemble strategies have been proposed often combining U-Net, Feature Pyramid Networks (FPN), and DeepLab variants to blend multi-scale features. These hybrid models improve stability in areas with complex textures, overlapping branches, or heterogeneous backgrounds. Collectively, the literature points toward growing interest in architectures that unify the speed of one-shot detection with the accuracy of fine-grained segmentation. This motivates exploration of models such as Unet, Ensemble strategies and YOLOv11, which are designed to provide stronger generalization, real-time performance, and higher-quality instance-level crown delineation.

## 4. Proposed Models and their Implementation
   
Our initial exploration of the literature strongly indicated that YOLO-based architectures tend to outperform classical semantic-segmentation models. However, in order to establish a clear baseline for comparison, we first implemented a U-Net segmentation model using EfficientNet-B3 as the encoder backbone, pre-trained on ImageNet. This baseline gave a mAP score of 0.1047, hence providing a meaningful reference point against which to evaluate the successive improvements obtained through the YOLOv11 segmentation pipeline.

##### Score of Our Baseline Model (U-Net)

<img width="614" height="71" alt="Screenshot 2025-12-03 at 11 26 05 PM" src="https://github.com/user-attachments/assets/150393f5-2aca-477c-b7d4-692ba5d80c63" />

### 4.1 Preprocessing 

Before training the final model, we designed a complete preprocessing workflow to handle the raw competition dataset (GeoTIFFs + polygon annotations).
Annotation preprocessing was a major component of this phase. The competition-supplied Solafune JSON files were parsed, validated, and converted into the COCO instance-segmentation format. During this transformation, every polygon was examined for structural integrity (e.g., non-self-intersecting geometry, correct vertex order, and adherence to image boundaries). Subsequently, the COCO annotations were programmatically converted into YOLO segmentation labels with normalized polygon coordinates. Finally, an 80/20 train–validation split was created, ensuring that both splits contained a representative mix of individual crowns, dense clusters, and complex urban scenes. 

### 4.2 Methodology 

Our methodology was built on training and evaluating multiple deep learning architectures before selecting the final model. Rather than adopting a single model from the outset, we pursued an iterative experimentation strategy that spanned classical segmentation networks, hybrid detection–segmentation systems, and modern object detectors. 

We began with the cleaned dataset produced from our preprocessing stage and proceeded through YOLO-based training. The training process employed YOLOv11s-seg as the primary architecture, using 640×640 images, AdamW optimization, and GPU-dependent batch sizes. Augmentation strategies like including flips, rotations, and multi-scale resizing were applied. YOLOv11’s improved mask head and decoupled detection–segmentation architecture contributed significantly to stable training and precise boundary reconstruction.

<img width="702" height="368" alt="Screenshot 2025-12-03 at 11 44 48 PM" src="https://github.com/user-attachments/assets/449e8d76-13b5-4282-a406-9757845b96ec" />

Model evaluation focused on mask mAP50 and mAP5095, providing a structured view of both coarse and fine-grained segmentation performance. Across all methods tested, YOLOv11 demonstrated the strongest ability to separate adjacent canopies, detect small trees, and maintain stability across epochs. While some limitations persisted, particularly regarding extremely small crowns, the model consistently displayed higher precision than recall, indicating a conservative prediction strategy that avoided excessive over-segmentation. This behavior aligned with our design preference for minimizing false positives in dense urban environments.
To support qualitative assessment, we implemented a custom visualization module that rendered predicted polygons overlaid on input imagery with confidence-weighted opacity. These tools proved essential for debugging boundary errors, understanding model uncertainty, and refining hyperparameters based on visual inspection rather than metrics alone.

After extensive exploration of U-Net variants, DeepLabV3+, EfficientNet-based FPNs, YOLOv8-seg, and multi-model ensembles, YOLOv11-seg emerged as the final model. It directly addressed the recurring failure modes documented earlier, offering improved cluster separation, superior sensitivity to small objects, refined boundary quality, and greater robustness to urban noise. This empirical performance was also consistent with the findings from recent high-resolution forestry literature, including the VHRTrees benchmark, which identified YOLO architectures as state-of-the-art for tree detection in VHR imagery, and the Ahmedabad case study, which demonstrated YOLO’s effectiveness in complex urban canopy environments.

## Results

The following table shows the metrics of our YOLOv11 Model:

<img width="546" height="269" alt="Screenshot 2025-12-03 at 11 33 11 PM" src="https://github.com/user-attachments/assets/81f8e538-dd8f-4458-97ce-69bf818f3d2f" />

The following table shows the details of all the other models explored by us:

<img width="377" height="288" alt="Screenshot 2025-12-03 at 11 48 57 PM" src="https://github.com/user-attachments/assets/690ced02-60fb-4b6d-9722-f3b5a033f11f" />


## Limitations and Conclusion 

Although YOLOv11 achieved the highest performance among all architectures tested, several limitations remained across data quality, model behavior, and computational constraints. The competition dataset contained imagery with resolutions varying from 10 cm to 80 cm, causing inconsistent feature granularity especially for small crowns that often spanned only a few effective pixels in lower-resolution tiles. This variability, combined with shadows, rooftops, and dense canopy clusters, led to systematically lower recall, reflecting the model’s conservative predictions and occasional under-segmentation in complex urban scenes. Training was further constrained by GPU memory limits, restricting batch size and reducing the breadth of possible augmentation and hyperparameter tuning. Additionally, polygon-level conversion into YOLO’s mask format introduced geometric quantization that may have contributed to modest mAP75 scores, especially in cases requiring highly precise boundary delineation.

Despite these limitations, the study demonstrates that YOLOv11-Seg is a strong and reliable architecture for real-time individual tree crown detection, significantly outperforming classical semantic-segmentation baselines such as U-Net, DeepLabV3+, FPN variants, and multi-model ensembles. With a final competition score of 0.25859 and a ranking of 58 out of 500 teams, the model showed robust generalization across diverse geographic contexts and strong separation of adjacent crowns, aligning with recent findings in high-resolution forestry research. The iterative experimentation, careful preprocessing workflow, and consistent improvements over baseline models highlight YOLOv11’s suitability for large-scale operational canopy mapping. Future advancements such as improved resolution datasets, transformer-enhanced decoders, or boundary refinement modules may further elevate performance, but the current results affirm YOLOv11 as an effective and practical solution for modern tree canopy segmentation tasks.

## References 
Ji, Z., Xu, J., Yan, L., Ma, J., Chen, B., Zhang, Y., Zhang, L., & Wang, P. (2024). Satellite remote sensing images of crown segmentation and forest inventory based on BlendMask. Forests, 15(8), 1320. https://doi.org/10.3390/f15081320

Morgan, G. R., Zlotnick, D., North, L., Smith, C., & Stevenson, L. (2024). Deep Learning for Urban Tree Canopy Coverage Analysis: A comparison and Case Study. Geomatics, 4(4), 412–432. https://doi.org/10.3390/geomatics4040022

Singla, J. G., Jaiswal, G., Space Applications Centre (ISRO), & HMR institute of Technology and Management, Delhi. (2021). Tree level change detection over Ahmedabad city using very high resolution satellite images and Deep Learning. In EarthArXiv [Journal-article].

Zheng, J., Yuan, S., Li, W., Fu, H., Yu, L., School of Artificial Intelligence, Ministry of Education Key Laboratory for Earth System Modeling, School of Geospatial Engineering and Science, National Supercomputing Center in Wuxi, Tsinghua University, & Tsinghua University (Department of Earth System Science)- Xi’an Institute of Surveying and Mapping Joint Research Center for Next-Generation Smart Mapping. (2023). A review of individual tree crown detection and delineation from optical remote sensing images. arXiv.


## VIDEO SUBMISSION
https://youtu.be/Iem80sYmcCc



