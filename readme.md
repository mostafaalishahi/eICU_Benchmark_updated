# eICU Benhmark updated

## Reference

**[Benchmarking machine learning models on multi-centre eICU critical care dataset](https://arxiv.org/abs/1910.00964v3)** by [Seyedmostafa Sheikhalishahi](https://scholar.google.it/citations?user=ck5btLoAAAAJ) and [Vevake Balaraman](https://scholar.google.it/citations?user=GTtAXeIAAAAJ) and [Venet Osmani](https://venetosmani.com/research/)

Relevant citation to our paper - [eICU paper](https://www.nature.com/articles/sdata2018178) by Tom J. Pollard et. al.

If you use this code or these benchmarks in your research, please cite the following publication.

```text
@article{sheikhalishahi2019benchmarking,
  title={Benchmarking machine learning models on multi-centre eICU critical care dataset},
  author={Sheikhalishahi, Seyedmostafa and Balaraman, Vevake and Osmani, Venet},
  journal={arXiv preprint arXiv:1910.00964},
  year={2021}
}
```

## Requirements

You must have the csv files of eICU on your local machine

### Packages

* numpy==1.15.0
* scipy==1.2.0
* scikit-learn==0.21.2
* pandas==0.24.1

For Feedforward Network and LSTM:

* Keras==2.2.4

## Structure

The content of this repository can be divide into two parts:

* data extraction
* running the models (baselines, LSTM)

## How to Build this benchmark

### Data extraction

Here are the required steps to create the benchmark. The eICU dataset CSVs should be available on the disk.

1. Clone the repository.

> git clone <https://github.com/mostafaalishahi/eICU_Benchmark_updated.git>
> cd eICU_Benchmark_updated

2. The following command generates one directory per each patient and writes patients demographics into pats.csv, the items extracted from Nursecharting into nc.csv and the lab items into lab.csv and then converts these three csv files into one timeseries.csv for each patient.
you will have one csv file with all the patients data in a time-series manner for all the four tasks.

> python data_extraction_root.py --eicu_dir "directory of csv files" output_dir "directory to save the extracted data"

### Run the models

3. Before going to run the experiment you need to set the desired configuration in the bash.py file (e.g. which tasks to choose with with settings)

4. All the desired settings for the training experiments are in the config.py file if you wish to change something.

5. The experiments are done using the following command, arguments related to task, numerical, categorical, artificial neural networks, one-hot encoding, and mortality window data. Those arguments can be provided as binary and for mortality window we consider the first 24 and 48 hours of the admission data.

6. The experiments can be run using

> python bash.py

