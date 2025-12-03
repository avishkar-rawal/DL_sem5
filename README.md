YOLOv11-Based Segmentation for Tree Canopy Mapping
1. Introduction

Tree canopy segmentation refers to the automated identification and delineation of individual or clustered tree crowns from high-resolution aerial or satellite imagery. Although widely applied across environmental and geospatial domains, existing segmentation methods often struggle to generalize across regions, imaging conditions, and mixed urban–rural landscapes. Variations in lighting, canopy density, background textures, and structural noise make consistent model performance challenging, emphasizing the need for more robust and adaptable approaches. To encourage progress in this area, Solafune introduced a dedicated tree-canopy-detection competition that brings together diverse aerial and satellite imagery and provides a rigorous benchmark for evaluating modern segmentation models. As part of this challenge, we reviewed a range of classical and contemporary architectures and ultimately selected YOLOv11, the current state of the art in real-time segmentation, for its strong performance and generalization capabilities.

2. Dataset description

The competition dataset is constructed from a curated collection of imagery sourced from SWISSIMAGE, NIAP, and national New Zealand aerial datasets. It includes high-resolution RGB TIFF images spanning both rural and urban environments, with spatial resolutions ranging from 10 cm to 80 cm. Polygon-annotated tree canopy masks are provided for the training split, accompanied by 150 evaluation images used for model validation. The dataset reflects real-world complexity, containing varied canopy structures, shadows, rooftops, mixed vegetation, roads, and other forms of environmental noise, making it a representative foundation for developing tree canopy segmentation models.

The polygon-based canopy annotations provided in the dataset use standard evaluation metric mean Average Precision (mAP-75) score.

