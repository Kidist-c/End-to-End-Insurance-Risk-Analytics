v📊 End-to-End Insurance Risk Analytics & Predictive Modeling

## Project Overview

This project is part of the AlphaCareInsurance Solutions(ACIS) analaytics intiative

- The goal is to analyze historical insurance Claim in SouthAfrica to :
  _ understand risk and profitability patterns
  _ Identify low risk Clients for targeted Premium \* Build Predictive models for dynamic ,risk-based insurance Pricing
- This README Summarizes the work Compeletd So far
  📁 Project Structure
  INSUARANCERISKANALYSIS-WEEK3/
  │--.github/workflows
  |--unitests.yml
  ├── data/ # Raw and processed datasets
  │ ├── insurance.txt # Original .txt dataset
  │ └── processed/ # Preprocessed dataset (generated)
  │--notebooks/
  ├── src/ # Source code modules
  │ ├── data_loader.py # Functions to load raw insurance data
  │ └── preprocessing.py # Preprocessing module for cleaning and feature engineering
  │
  ├── tests/ # Test modules
  │ └── test_preprocessing.py # Tests for preprocessing and data integrity
  │
  ├── requirements.txt # Python dependencies
  ├── README.md # Project overview and instructions
  └── .gitignore # Git ignore file
  🔧 Setup Instructions
  1.Clone Github Repository
  - git clone https://github.com/Kidist-c/InsuarnceRiskAnalysis-week3.git
- cd InsuarnceRiskAnalysis-week3
  2.Create Virtual Enviroment and Activate it
  - python -m venv venv #create virtual enviroment
  - venv\Scripts\activate
    3.install Dependencies
    pip install -r requirements.txt
    📝 Work Done So Far
    1.Gitsetup
  - Created a repo - added .gitignore file - created branch "task-1"
    2.Data Loading
  - loaded Insurance.txt dataset using custom data_loader module
    3.Preprocessed Data
  - droped columns with excessive missing values(CustomValueEstimate","CapitalOutstanding",
    "WrittenOff","Rebuilt","Converted","CrossBorder","NumberOfVehiclesInFleet")
  - Filled missing Value:
  - catagorical with -mode
- numerical with-Median
  -Boolen with-Mode
  4.Testing
- Added pytest-based placeholder tests:

- Dropped columns

  - filling missing values
    5,CI/CD:

- Setup GitHub Actions workflow (python-ci.yml) to:

      * Checkout the repo

      * Set up Python

       * Install dependencies

       * Run pytest tests automatically on every push or pull request

### 📊 Exploratory Data Analysis (EDA)

The goal of the Exploratory Data Analysis (EDA) stage is to understand the structure, quality, and key insights in the processed insurance dataset. This step helps reveal data patterns, highlight anomalies, and guide further modeling decisions.

- Data Summarization:

* Described the shape , the dtypes and the missing values of the data set
* UNIVARIANTE ANALYSIs
* BIVARINTE ANALYSIS
* CORRELATION MATRIX
* INSIGHT PLOT
  📦 DVC Setup Guide (Data Version Control)

This project uses DVC (Data Version Control) to manage large datasets without pushing them to GitHub. Below is the exact setup process I followed.

✅ 1. InstallED DVC
pip install dvc

✅ 2. InitializeD DVC in the Project

dvc init

This creates the .dvc/ directory and updates .gitignore to keep large files out of Git.

✅ 3. Set Up Local Remote Storage
Created a folder on my machine to store dataset versions:
mkdir path/D:/dvc_storage

Added the directory as a DVC remote:
dvc remote add -d localstorage D:/dvc_storage

dvc remote add -d localstorage path/to/local_storage

✅ 4. Track My Dataset with DVC

Placed my dataset in the project directory, then run:

dvc add insurance.txt

This creates a data.csv.dvc file that Git will track.

✅ 5. Commited the DVC Metadata
git add insurance.txt.dvc .gitignore
git commit -m "Track dataset using DVC"

✅ 6. Pushed Dataset to the Local Remote Storage
dvc push
