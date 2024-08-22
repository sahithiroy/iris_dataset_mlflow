import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()  # Load the dataset from sklearn's built-in datasets
X = iris.data  # Features of the dataset
y = iris.target  # Target labels of the dataset

# Set the MLflow tracking URI to a local MLflow server (commented out)
# This is typically where you'd specify the MLflow server address
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Configure MLflow to use MySQL as the backend store
mlflow.set_tracking_uri("mysql://root:sahithi@localhost/metrics")
# This connects MLflow to a MySQL database where it will log all experiment data

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 80% of the data will be used for training and 20% for testing

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Initialize a RandomForestClassifier with 100 trees and a fixed random seed for reproducibility

# Start an MLflow run to log this experiment
with mlflow.start_run():
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    predictions = model.predict(X_test)
    
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, predictions)
    
    # Log model parameters to MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    # Log the number of estimators and the random seed used in the model
    
    # Log the accuracy metric to MLflow
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the trained model to MLflow
    mlflow.sklearn.log_model(model, "random_forest_model")
    # This stores the model in the experiment's artifact store under the specified name
    
    print(f"Model accuracy: {accuracy:.4f}")
    # Output the accuracy of the model
