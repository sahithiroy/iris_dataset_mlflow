# MLflow Iris Classification
### This project demonstrates how to train a RandomForestClassifier on the Iris dataset and log the model, parameters, and metrics using MLflow. The experiment data is stored in a MySQL database.

## Prerequisites
Ensure that you have the following installed on your system:
```
Python 3.8 or higher
MySQL
Python packages (specified in requirements.txt)
Installation
```
Clone the repository:
```
git clone https://github.com/yourusername/mlflow-iris-classification.git
cd mlflow-iris-classification
```

Set up a Python virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

```
Install the required Python packages:

```
pip install -r requirements.txt
```
Set up MySQL:

Create a new MySQL database named metrics:

```
CREATE DATABASE metrics;
```
Create a user and grant permissions (if needed):

```
CREATE USER 'mlflow_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON metrics.* TO 'mlflow_user'@'localhost';
FLUSH PRIVILEGES;
```
Update the mlflow.set_tracking_uri in the code if using a different username/password.

## Running the Project
### Run the MLflow experiment:

```
python iris_mlflow.py
```
This will train the RandomForestClassifier on the Iris dataset, log the model parameters, metrics, and the model itself to MLflow, and store the data in the MySQL database.

View the MLflow UI (if you're using an MLflow tracking server):

If you have an MLflow server running, you can view the experiments by visiting:

```
http://127.0.0.1:5000
```
Alternatively, you can start an MLflow UI locally to visualize the logged runs:

```
mlflow ui
```
Then, open your web browser and go to http://127.0.0.1:5000.

## Directory Structure
```
mlflow-iris-classification/
│
├── iris_mlflow.py            # The main Python script to run the experiment
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── venv/                     # Virtual environment directory (optional)
```
## Troubleshooting
ModuleNotFoundError: Ensure that you have activated your virtual environment and installed all dependencies from requirements.txt.

MySQL Connection Issues: Double-check your MySQL server settings, ensure it’s running, and the credentials in mlflow.set_tracking_uri are correct.

License
This project is licensed under the MIT License. See the LICENSE file for details.

## Notes:
Replace https://github.com/yourusername/mlflow-iris-classification.git with the actual GitHub repository URL.
Ensure requirements.txt includes all necessary Python packages, like mlflow, scikit-learn, mysqlclient, etc.
If you’re using a different username or password for MySQL, update the instructions accordingly.





