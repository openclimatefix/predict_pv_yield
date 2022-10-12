# Ablation Results

Want to investigate the influence of the following input data

- satellite data
- NWP
- GSP history
- PV history
- sun/time features

## Removing one input

The idea is to keep all input data, and remove just one to see the effect
We calculate the delta between the run and a reference run where all data inputs are used. 
If the delta is small, then this input data is not important. 

| Data input      | Loss (MAE Exp)| Experiment | Delta | Influence % |
| ----------- | ----------- | ----- | --- | --- |
| Satellite      | 0.0318       |1210 |  0.009 | 14
| NWP   | 0.0325        |1201 | 0.016 | 24
| GSP   |  0.0343       | 1205| 0.034 | 52
| PV   | 0.0312        | 1207 | 0.003 | 5
| Sun   | 0.0312       | 1208| 0.003 | 5
| ref   | 0.0309        | 1197 | 

## Only one data input

Idea is to remove all input data apart from one, and see the effect. 
We use 1/delta as if the delta is small, then the data is important.

| Data Input      | Loss (MAE Exp)| Experiment | 1 / Delta | Influence % |
| ----------- | ----------- | ----- | --- | --- |
| Satellite      |  0.0482      | 1211| 56.8 | 19
| NWP   |          0.0420|  1212| 87.7 | 30
| GSP   |         0.0396|  1213 | 111.1 | 38
| PV   |        0.0493 |  1214 | 20.3 | 7
| Sun   |    0.0642     | 1215| 15.6 | 6
| ref   | 0.0306        | 1197 |

## Average both methods

By averagin both of the above methods we get

| Data Input      | Influence % |
| ----------- | ----------- |
| Satellite      |  16
| NWP   |          27
| GSP   |        45
| PV   |       6  | 
| Sun   |    6    | |