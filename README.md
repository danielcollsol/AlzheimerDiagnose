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

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

No test required.

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Daniel Coll** - *Initial work* - [AlzheimerDiagnose](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
