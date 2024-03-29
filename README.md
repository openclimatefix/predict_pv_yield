# Intro
Early experiments on predicting solar electricity generation over the next few hours, using deep learning, satellite imagery, and as many other data sources as we can think of :)

These experiments are focused on predicting solar PV yield.

Please see [SatFlow](https://github.com/openclimatefix/satflow/) for complementary experiments on predicting the next few hours of satellite imagery (i.e. trying to predict how clouds are going to move!)

And please see [OCF's Nowcasting page](https://github.com/openclimatefix/nowcasting) for more context.

# Installation

From within the cloned `predict_pv_yield` directory:

```
conda env create -f environment.yml
conda activate predict_pv_yield
pip install -e .
```
