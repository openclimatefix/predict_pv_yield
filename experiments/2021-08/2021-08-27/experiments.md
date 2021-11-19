# Daily Experiments

Ran hydra for the first time, for hyper parameters optermization.
It did 2 full runs, then I think ran out of memory caused a funny error.
Now have install 'psutil' so that cpu and memory is logged to neptune.

https://app.neptune.ai/o/OpenClimateFix/org/predict-pv-yield/e/PRED-160/monitoring
Validation error after 10 epochs - 0.073

conv3d_channels = 32
fc1_output_features = 16
fc2_output_features = 128
fc3_output_features = 16

https://app.neptune.ai/o/OpenClimateFix/org/predict-pv-yield/e/PRED-161/monitoring
Validation error after 10 epochs - 0.073

conv3d_channels = 32
fc1_output_features = 32
fc2_output_features = 16
fc3_output_features = 16

https://app.neptune.ai/o/OpenClimateFix/org/predict-pv-yield/e/PRED-162/monitoring
Validation error after 2 epochs - 0.076 (then error happened in 3rd epoch)

conv3d_channels = 32
fc1_output_features = 64
fc2_output_features = 16
fc3_output_features = 8
