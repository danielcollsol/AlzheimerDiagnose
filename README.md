# ML Classifier for Alzheimer's Disease Diagnosis 

The early diagnosis of Alzheimer's disease (AD) plays an essential role in patient care
since it allows to take prevention measures, specially at the early stages, before
irreversible brain damages are shaped. Although many studies have applied deep learning
methods for computer-aided-diagnosis of AD obtaining significant results and high
prediction accuracies, the neuroimages utilized in these researches were preprocessed
and, in result, far different from those produced originally by the medical equipment. In this
study, a structural magnetic resonance imaging classifier based on a convolutional neural
network has been designed for the diagnosis of AD from non preprocessed neuroimages.
Keras pre-trained models Inception-v3 and VGG-16 have been implemented using transfer
learning techniques to classify AD from cognitive normal (CN) and mild cognitive
impairment (MCI), its prodromal stage. The proposed approach is validated on the
standardized structural MRI datasets from the Alzheimer’s Disease Neuroimaging Initiative
(ADNI) project. This approach achieves an accuracy of 62% and 36% on Inception-v3 and
55% and 38% on VGG-16 for AD vs CN and AD vs MCI vs CN classification respectively.
The further improvement and research of this algorithms using pre-selected 2,5 or 3D
images have the potential to provide a rapid, accessible and computational efficient
data-driven assessment of Alzheimer Disease.

## Getting Started

Python 3.7 procedures for each model written in jupyter notebooks.

### Prerequisites

Requirements in requirements.txt

### Installing

Used conda environment and conda install package.

## Deployment

Add additional notes about how to deploy this on a live system

## Author

* **Daniel Coll** - *Initial work* - [AlzheimerDiagnose](https://github.com/danielcollsol)


