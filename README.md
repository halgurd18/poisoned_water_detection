# poisoned_water_detection
This repository contains a dataset and machine learning algorithms to detect poisoned water from clean water via using equivalent Smartphone embedded Wi-Fi CSI data.

The machine learning algorithm (inclduing k-NN, SVM, LSTM, and Ensemble) are written in MATALB code!

The testbed is shown in img/p1.jpg

The equivalent Smartphone embedded Wi-Fi chipsets are shown in img/p2.jpg

The amplitude and phase measurements of the Wi-Fi CSI data is selected as vectors features!

Each of these vectors includes 64 feature values, i.e. the amplitude vector has 64 values and phase vector has 64 values.

The method provide accurate classification result starts from 82% (via k-NN) up to 92% (via Ensemble)
