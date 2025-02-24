# DNP-KD
## Introduction
Multi-class Anomaly Detection (AD) methods have recently gained significant attention. However, Knowledge Distillation-based (KD-based) AD methods as one of the mainstream AD approaches, exhibit considerably poorer performance in multi-class scenarios than in single-class settings. Our analysis indicates that, compared to single-class scenarios, the increased diversity and quantity of data in multi-class scenarios make the student's extracted anomaly representations more similar to those of the teacher. This leads to a degradation in the performance of KD-based AD methods, which rely on the discrepancy between the student's and teacher's representations to perform anomaly detection. Therefore, this paper proposes a Dedicated Normal Pattern Knowledge Distillation (DNP-KD) method for multi-class AD. Specifically, DNP-KD employs a nested encoder-decoder structure to eliminate redundant information from the teacher features. Within this framework, the hierarchical bottleneck layers retain only the critical information of normal pattern features, which is then delivered to the student network serving as the decoder. To further prevent anomalous information from flowing into the student network, we propose a Frequency-aware Feature Compression (FaFC) module. The FaFC module compresses global information by applying frequency-domain masking operations to the teacher's features. The student can decode features similar to the teacher's based on the compressed embeddings when encountering normal samples, but this has failed to be replicated with anomalous samples. Experiments conducted on two industrial datasets (MVTec-AD and VisA) and a medical dataset (BMAD) demonstrate that DNP-KD delivers competitive detection performance when compared to the current state-of-the-art AD multi-class methods.
## Overview of DNP-KD
![image](https://github.com/AutomanXXX/DNP-KD/blob/main/framework.png)
## Train
`python train.py`
## Test
Download the model checkpoints and extract the zip so that the checkpoints folder will be located in the base directory of this repository.
Download link:
[[[https://drive.usercontent.google.com/download?id=1XJ0U_jWXnVRzjfT_3SQPoU46GgLALtOa&export=download](https://drive.usercontent.google.com/download?id=1XJ0U_jWXnVRzjfT_3SQPoU46GgLALtOa&export=download)]

`python test.py`
