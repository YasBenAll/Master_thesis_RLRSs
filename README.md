# Master_thesis_RLRSs
 

## Overview



## Project Structure

momdp_project/  
├── main.py  
├── momdp/  
│   ├── \_\_init\_\_.py  
│   ├── momdp.py  
│   ├── policy.py  
│   └── evaluation.py  
├── data/  
│   ├── \_\_init\_\_.py  
│   └── gen_dataset.py  
├── tests/  
│   ├── \_\_init\_\_.py  
│   ├── test_momdp.py  
│   ├── test_policy.py  
│   └── test_evaluation.py  
└── requirements.txt  


### Files and Directories

- `main.py`: Entry point of the project. Initializes and runs the MOMDP system.
- `momdp/`: Contains the core modules for the MOMDP system.
  - `momdp.py`: Defines the MOMDP class and its methods.
  - `policy.py`: Contains functions related to policy creation.
  - `evaluation.py`: Contains functions for evaluating the MOMDP.
- `data/`: Contains data handling modules.
  - `item_popularity.py`: Functions to get item popularity data.
- `tests/`: Contains unit tests for the project.
  - `test_momdp.py`: Tests for the MOMDP class.
  - `test_policy.py`: Tests for policy creation.
  - `test_evaluation.py`: Tests for evaluation functions.
- `requirements.txt`: Lists the dependencies required for the project.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/YasBenAll/Master_thesis_RLRSs.git
    cd Master_thesis_RLRSs
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main script to initialize and evaluate the MOMDP system:

```bash
python main.py
