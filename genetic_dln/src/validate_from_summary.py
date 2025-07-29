import os
from genetic_dln.src.validation.ga_run_summary_validator import validate_run_summary
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
print("Current dir: {}".format(current_dir))

# DEPENDS ON YOUR PATHS, MIGHT NEED CHANGING
logs_dir = os.path.join(current_dir, "logs") #where the ga_logs are stored

summary_dir = os.path.join(current_dir, "..", "..", "ga_results") #where the summary (referencing the above logs) is stored

summary_file = "ga_runs_summary_20250515_082617.json"

print(f"Log directory absolute path: {os.path.abspath(logs_dir)}")

def run_validate_from_summary():
    """
    Executes the validation process for a given run summary and saves the validation summary.
    """
    # Define the paths

    print("Log dir: {}".format(logs_dir))
    print("Summary dir: {}".format(summary_dir))

    run_summary_path = os.path.join(summary_dir, summary_file)
    output_dir = os.path.join(summary_dir, "val")

    # Ensure directories exist
    if not os.path.exists(run_summary_path):
        raise FileNotFoundError(f"Run summary file not found: {run_summary_path}")
    if not os.path.exists(logs_dir):
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Validate the run summary with progressive saving
        validate_run_summary(run_summary_path, logs_dir, output_dir)
        print(f"Validation process completed successfully. Progress saved incrementally.")
    except Exception as e:
        print(f"Validation failed: {e}")


if __name__ == "__main__":
        run_validate_from_summary()