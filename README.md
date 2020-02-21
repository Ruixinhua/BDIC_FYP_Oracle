# BDIC FYP Ocale
This project is the final year project of BDIC students

* [BDIC FYP Ocale](#BDIC FYP Ocalet)
    * [Members](members)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)


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