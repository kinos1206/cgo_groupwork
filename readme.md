# cgo-groupwork
## Requirement
- Python 3.11.4
- matplotlib 3.7.1
- pandas 2.2.2
- Pytorch 2.0.1
- torchvision 0.15.2

## Dataset
The mnist dataset will be saved in the directory `./data`. 
## Training
To train a model:
``` bash
python3 main.py --epoch_limit 100 --epoch_min 5
```  
or you can directly:
``` bash 
$ bash main.sh
```  
You can check optimised model in the file `./model` and the logs in the file `./logs`.
Runtime options will be saved in the file `./model/option.txt`.  More training flags in the files `./options.py`.
## Acknowledgments
Part of the code is based upon [PyTorch MNIST](https://qiita.com/takawamoto/items/42ff569be496621fc016).
