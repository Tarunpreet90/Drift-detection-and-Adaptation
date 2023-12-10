# Drift-detection-and-Adaptation
The various drift detection methods such as PSI, KS, WDM,
CVM, MWT were used to identify concept drift in incoming
batch of datasets. For experimental evaluation, the HR and IBP
time series data of 945 patients undergoing cardiac surgeries
were utilized as the historical dataset, while the batch of 100
new patients undergoing the same surgery was used as the
test dataset. A virtual drift was induced in the test datasets
by changing the data distribution of outcome values. 

 In the realm of medical data, accurately predicting outcomes like ICU and hospital stays following cardiac surgeries becomes essential. The study underscores the importance of adapting to concept drift in such scenarios. To address the issue of drift adaptation, three dynamic strategies—adaptive learning, incremental learning, and ensemble learning techniques have been utilized. Adaptive learning involves retraining models on modified datasets after drift detection, while incremental learning continually updates models as new data arrives. Ensemble learning, on the other hand, combines models to handle recurring drift and better adapt to changing data patterns.
