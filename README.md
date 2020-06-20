aqi_thailand2
==============================

# Directory Tree

aqi_thailand2 
├── data: raw data and processed data for each cities 
├── docs: code documentations  
├── models: model files for each city
│   ├── bangkok
│   ├── chiang_mai
│   └── hanoi
├── notebooks: generate figure and experiment with codes
├── reports
│   ├── chiang_mai_report.docx (remove from public)
│   ├── figures : figure for the reports
├── src
│   ├── data: function for downloading and cleaning up raw data 
│   ├── features:  Dataset object is responsible for putting raw data together, feature engineering and preparing matricies for model 
│   ├── models: model builder and hyperparameter searcher
│   ├── visualization: plotting data 
│   ├── imports.py: import python packages
└── gen_functions.py: functions not fit in the above folders
