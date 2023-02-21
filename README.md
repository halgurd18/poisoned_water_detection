# poisoned_water_detection
This repository contains a dataset and machine learning algorithms to detect poisoned water from clean water via using equivalent Smartphone embedded Wi-Fi CSI data.

The machine learning algorithm (including k-NN, SVM, LSTM, and Ensemble) are written in MATLAB code!

The testbed is shown in img/p1.jpg

The equivalent Smartphone embedded Wi-Fi chipsets are shown in img/p2.jpg

The amplitude and phase measurements of the Wi-Fi CSI data are selected as vector features!

Each of these vectors includes 64 feature values, i.e. the amplitude vector has 64 values, and the phase vector has 64 values.

The method provides accurate classification results starting from 82% (via k-NN) up to 92% (via Ensemble)

This research and dataset is supported and created by a group of researchers from Koya University, and you can read more on:
https://koyauniversity.org/ku/node/457

If anyone want to use the dataset and make further research, kindly, should cite the paper as the following:
<br> 
Maghdid, H.S., Salah, S.R.H., Taher, A.H., Bayram, H.M., Sabir, A.T., Kaka, K.N., Taher, S.G., Abdulrahman, L.S., Al-Talabani, A.K., Asaad, S.M. and Asaad, A., 2023. A Novel Poisoned Water Detection Method Using Smartphone Embedded Wi-Fi Technology and Machine Learning Algorithms. arXiv preprint arXiv:2302.07153.
