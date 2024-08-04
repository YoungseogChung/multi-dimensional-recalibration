# Sampling-based Multi-dimensional Recalibration

Repository for [Sampling-based Multi-dimensional Recalibration](https://proceedings.mlr.press/v235/chung24a.html) (Published at ICML 2024)

## Installation 

We recommend creating a designated conda environment and installing the specified verions of the required packages by running the following command:
```
conda env create -f environment.yml
```
This will create a conda environment called "multidim_recal".

## Example Usage

To run the code, activate the conda environment and simply run main.py in the root directory:
````bash
conda activate multidim_recal
python main.py
````

The code will run SD recalibration (single dimensional recalibration) and HDR recalibration (highest density region recalibration) on the test predictions and data provided in `data_predictions`.
`data_predictions` contains the base model predictions and data for the test split on the `scpf` dataset across all 5 random seeds. 

After the recalibration procedure is finished, the sample-based metrics will be computed for the methods SD recalibration, HDR recalibration, and Prehoc.
The average results across the 5 seeds will be printed to stdout.

## Citation
This reposity was based on the following paper:
```
@InProceedings{pmlr-v235-chung24a,
  title = 	 {Sampling-based Multi-dimensional Recalibration},
  author =       {Chung, Youngseog and Char, Ian and Schneider, Jeff},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {8919--8940},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/chung24a/chung24a.pdf},
  url = 	 {https://proceedings.mlr.press/v235/chung24a.html},
  abstract = 	 {Calibration of probabilistic forecasts in the regression setting has been widely studied in the single dimensional case, where the output variables are assumed to be univariate. In many problem settings, however, the output variables are multi-dimensional, and in the presence of dependence across the output dimensions, measuring calibration and performing recalibration for each dimension separately can be both misleading and detrimental. In this work, we focus on representing predictive uncertainties via samples, and propose a recalibration method which accounts for the joint distribution across output dimensions to produce calibrated samples. Based on the concept of highest density regions (HDR), we define the notion of HDR calibration, and show that our recalibration method produces samples which are HDR calibrated. We demonstrate the performance of our method and the quality of the recalibrated samples on a suite of benchmark datasets in multi-dimensional regression, a real-world dataset in modeling plasma dynamics during nuclear fusion reactions, and on a decision-making application in forecasting demand.}
}
```

## Questions?

Please do not hesitate to reach out if you have any questions: [Youngseog Chung](https://github.com/YoungseogChung) (youngsec (at) cs.cmu.edu)