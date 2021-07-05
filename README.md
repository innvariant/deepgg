# DeepGG: Deep Graph Generator
Learning a state-based generative model of graph distributions.
![Sample of generated graphs from DeepGG](res/sample.png)

```bibtex
@inproceedings{stier2020deep,
  title={DeepGG: a Deep Graph Generator},
  author={Stier, Julian and Granitzer, Michael},
  booktitle={Advances in Intelligent Data Analysis XIX: 19th International Symposium on Intelligent Data Analysis, IDA 2021, Porto, Portugal, April 26--28, 2021, Proceedings},
  pages={325},
  organization={Springer Nature}
}
```

# Reproducing Experiments
- install the conda environment with ``conda env create -f environment.yml``
- activate the environment ``conda activate sur-deepgg``
- configure your (hyper)parameters (first ~20 variables)
- invoke as much as possible computations via ``python deepgg_pipeline.py``
- merge the computations as shown in *deepgg-merge-computations.ipynb*
- have a look over the exemplary notebooks of how to visualize some aspects of the computed models
