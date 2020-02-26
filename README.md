# BDIC FYP Oracle
This project is the final year project of BDIC students

* [BDIC FYP Oracle](#BDIC-FYP-Oracle)
    * [Members](members)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Customization](#customization)
	* [Base class](#base-class)


## Members
- Interns: Dairui Liu, Xiuxian Li, Gechuan Zhang, Jie Lei
- Supervisor: Ruihai Dong, Yu An

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4 (1.4 recommended)
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization][#tensorboardx-visualization])

## Features
* Clear folder structure which is suitable for many deep learning projects.
* `.json` config file support for convenient parameter tuning.
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and more.
  * `BaseDataLoader` handles batch generation, data shuffling, and validation data splitting.
  * `BaseModel` provides basic model summary.
  * `BaseDataset` define the operation about how to get data from dataset
  
## Folder Structure
  ```
  BDIC_FYP_Oracle/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── new_project.py - initialize new project with template files
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── dataset/ - default directory for storing input data
  │   ├── single_character/ - storing single character image data
  │   ├── oracle_dataset.py - define OracleDataset class here
  │   └── ... - add your data and dataset class here
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```
  
## Customization

### Project initialization
Use the `new_project.py` script to make a new project directory with template files.
`python new_project.py ../NewProject` then a new project folder named 'NewProject' will be made. 
If the new project is totally different from this one, it should be put in another folder(using ../ return last layer).
This script will filter out unneccessary files like cache, git files or readme file.

### Custom CLI options

Changing values of config file(in `.json` format) is a clean, safe and easy way of tuning hyperparameters. However, sometimes
it is better to have command line options if some values need to be changed too often or quickly.

### Loss
Custom loss functions can be implemented in 'model/loss.py'. Use them by changing the name given in "loss" in config file, to corresponding name.

#### Metrics
Metric functions are located in 'model/metric.py'.

Monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```json
  "metrics": ["accuracy", "top_k_acc"],
  ```
  
### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

### Checkpoints
Specify the name of the training session in config files:
  ```json
  "name": "Oracle_LeNet",
  ```

The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```

## Base class

### BaseDataLoader
- `BaseDataLoader` is a subclass of `torch.utils.data.DataLoader`, so directly use DataLoader is just OK.
    `BaseDataLoader` handles:
    * Generating next batch
    * Data shuffling
    * Generating validation data loader by calling
      `BaseDataLoader.split_validation()`
- **DataLoader Usage**
`BaseDataLoader` is an iterator, to iterate through batches:
  `BaseDataLoader` is an iterator, to iterate through batches:
  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```
    * **Example**

- **Inherit ```BaseDataLoader``` to create a new DataLoader**
    * specific dataset object when initial the object(create a Dataset class first, see:https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
    * initial an instance of DataLoader when training and testing:
    ```python
    # training setup data_loader instances
    data_loader = config.init_obj('data_loader', new_data_loader)
    # testing setup data_loader instances
    data_loader = getattr(new_data_loader, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )
    ```
    
### BaseModel
* **Writing a new model**
1. **Inherit `BaseModel`**

    `BaseModel` handles:
    * Inherited from `torch.nn.Module`
    * `__str__`: Modify native `print` function to prints the number of trainable parameters.

2. **Implementing abstract methods**

    Implement the foward pass method `forward()`

* **Example**

  Please refer to `model/oracle_model.py` for a LeNet example.
  
### BaseDataset
* **Build a custom Dataset class**
1. **Inherit `BaseDataset` or `torch.utils.data.Dataset
    `BaseDataset` handles:
    *  Inherited from `torch.nn.data.Dataset`
    *  `____getitem__`: Define the operation when get an item from dataset
2. **Implementing `__len__` and `__getitem__` method

### BaseTrainer
* **Define training process**
1. **Implementing `_train_epoch` method
    * Write training logic for an epoch
    * Return a log that contains average loss and metric in this epoch
 