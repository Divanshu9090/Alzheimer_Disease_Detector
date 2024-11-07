# Alzheimer_Disease_Detector
# Introduction
This project aims to build a deep learning model to detect Alzheimer's disease with cross-sectional MRI data (more than 86,000 MRI slice images) using 2D-convoluted neural networks (CNN). Normalization and regularization techniques were used to create a stable model with good generalization performance. Despite its shallow architecture, the preliminary model developed here (SCNN4) was able to achieve an accuracy of 99.9%, a significant improvement from the current deep learning models. The training time at the moment is 18 minutes using a T4 GPU provided by Google Colab.

# Future work
Currently, the model is trained without considering class imbalance. In addition, images were split for training, validation and testing randomly without considering subject identity. The former issue may lead to a biased model (see the section "Exploring the dataset") and the latter may impact the model's generalization performance due to data leakage. These issues will need to be addressed in the future. In terms of performance metrics, other metrics besides accuracy will be needed to assess the model more comprehensively, such as the model's prediction precision and generalization performance.

# Data sourcing
The training dataset was compiled by the Open Access Series of Imaging Studies (OASIS) (Marcus et al., 2007). The derived dataset was partially preprocessed by Ninad Aithal (Centre for Brain Research IISc - Bangalore, IN), including .nii to .jpg file conversion and patient classification (see this page for more information).
