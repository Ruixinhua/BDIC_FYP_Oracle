# BDIC FYP Ocale
This project is the final year project of BDIC students

* [BDIC FYP Ocale](#BDIC FYP Ocalet)
    * [Members](members)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Customization](#customization)


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

  Please refer to `model/mnist_model.py` for a LeNet example.