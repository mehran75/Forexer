# Forexer
RNN based forex predictor

# Requirements
* Pytorch ([Install Pytorch and cuda](https://pytorch.org/get-started/locally/))
* Numpy ([Install](https://numpy.org/install/))
* Pandas ([Install](https://pandas.pydata.org/getting_started.html))
* Sickit-learn ([Install](https://scikit-learn.org/stable/install.html))
* Downloading dataset for training ([Download](https://www.dukascopy.com/plugins/fxMarketWatch/?historical_data))


# Execute this repo
The easiest way is to run it via makefile:
```bash
git clone https://github.com/mehran75/Forexer.git
cd Forexer
conda activate <env name> # if you have conda installed on your machine
make run 
```
output:
```
python main.py "configuration/parameters_eurusd.yml"

********************************************************
                         Forexer
********************************************************


GPU is available
GPU device count: 1
configuration/parameters_eurusd.yml
Loaded dataset/EURUSD/EURUSD_Candlestick_1_M_BID_01.01.2018-21.11.2020.csv data with shape of (1516311, 6)
Training the model...
 epoch:   1/ 10, loss: 0.00138055, completed: 100.00%

 epoch:   2/ 10, loss: 0.00128420, completed: 39.88%
```
Note that you have to register in [truefx.com](http://truefx.com/) and put your username, and password in the `makefile`
at the root of repository.

  


### If pytorch didn't recognize cuda
It could be solve by:
```bash
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```
