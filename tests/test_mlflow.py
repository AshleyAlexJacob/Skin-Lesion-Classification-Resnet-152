import os
import time
import mlflow
from dotenv import load_dotenv

def test_mlflow_connection():
    # Load environment variables from .env
    load_dotenv()
    
    # Check if tracking URI is set
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri or "placeholder" in tracking_uri:
        print("Error: MLFLOW_TRACKING_URI is not set or still has placeholder in .env file.")
        print("Please update your .env file with your DagsHub credentials.")
        return
        
    print(f"Testing connection to: {tracking_uri}")
    
    # Set experiment
    experiment_name = "mlflow_connection_test"
    mlflow.set_experiment(experiment_name)
    
    print(f"Starting run in experiment: {experiment_name}")
    try:
        with mlflow.start_run():
            # Log some dummy parameters
            mlflow.log_params({
                "test_param_1": "value_1",
                "test_param_2": 42
            })
            print("Logged test parameters...")
            
            # Log some dummy metrics
            for i in range(5):
                mlflow.log_metric("dummy_loss", 1.0 / (i + 1), step=i)
                mlflow.log_metric("dummy_acc", min(100.0, 50.0 + (i * 10)), step=i)
                time.sleep(0.5)
            print("Logged dummy metrics over 5 steps...")
            
            # Create and log a dummy artifact
            with open("dummy_artifact.txt", "w") as f:
                f.write("This is a test artifact to ensure file logging works.")
            
            mlflow.log_artifact("dummy_artifact.txt")
            print("Logged dummy artifact...")
            
            # Clean up local artifact file
            os.remove("dummy_artifact.txt")
            
        print("\nSuccess! MLflow connection and logging is working perfectly.")
        print("You can verify the run in your DagsHub MLflow UI.")
        
    except Exception as e:
        print(f"\nError during MLflow logging: {e}")

if __name__ == "__main__":
    test_mlflow_connection()
