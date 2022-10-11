# Ablation Results

Want to investigate the influence of the following input data

- satellite data
- NWP
- GSP history
- PV history
- sun/time features

## Removing one input

The idea is to keep all input data, and remove just one to see the effect

| Data input      | Loss (MAE Exp)| Experiment | Delta | Influence % |
| ----------- | ----------- | ----- | --- | --- |
| Satellite      | 0.0318       |1210 |  0.009 | 14
| NWP   | 0.0325        |1201 | 0.016 | 24
| GSP   |  0.0343       | 1205| 0.034 | 52
| PV   | 0.0312        | 1207 | 0.003 | 5
| Sun   | 0.0312       | 1208| 0.003 | 5
| ref   | 0.0309        | 1197 | 

## Only keeping one

Idea is to remove all input data apart from one, and see the effect

| Data Input      | Loss (MAE Exp)| Experiment | Delta | Influence % |
| ----------- | ----------- | ----- | --- | --- |
| Satellite      |        | | 
| NWP   |         | |  |
| GSP   |        | |  |
| PV   |         |  |
| Sun   |         | |
| ref   | 0.0306        | 1197 |