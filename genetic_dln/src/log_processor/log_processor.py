import json
import os

class LogProcessor:
    """
    A utility class for processing genetic algorithm logs.
    """

def extract_best_individual(ga_log_path):
    """
    Extract the best individual from a given ga_log.

    Args:
        ga_log_path (str): Path to the ga_log file.

    Returns:
        best_individual, ga_log_path
        best_individual (str): json-formatted entry in the ga_log file.
        ga_log_path (str): Path to the ga_log file.
    """
    if not os.path.exists(ga_log_path):
        raise FileNotFoundError(f"The specified log file does not exist: {ga_log_path}")

    # Load the ga_log JSON file
    with open(ga_log_path, "r") as f:
        ga_log = json.load(f)

    # Ensure the 'generations' key exists in the log
    if "generations" not in ga_log or not ga_log["generations"]:
        raise ValueError(f"The log file does not contain valid generations: {ga_log_path}")

    # Get the last generation
    last_generation = ga_log["generations"][-1]

    # Ensure the last generation contains fitness data
    if "fitness_data" not in last_generation or "best_individual" not in last_generation["fitness_data"]:
        raise ValueError(f"Fitness data missing in the last generation of log: {ga_log_path}")

    # Extract the best individual
    best_individual = last_generation["fitness_data"]["best_individual"]

    # Return the best individual with the log path for mapping
    return best_individual, ga_log_path

def extract_best_from_run_summary(runs_summary_path, logs_dir):
    """
    Enhances a ga_runs_summary by adding the best individual for each run.

    Args:
        runs_summary_path (str): Path to the ga_runs_summary file.
        logs_dir (str): Directory containing the GA log files.

    Returns:
        list[dict]: Updated summary structure where each run entry contains:
            - Original fields from the summary.
            - 'best_individual': The best individual extracted from the log.
    """
    if not os.path.exists(runs_summary_path):
        raise FileNotFoundError(f"Runs summary file not found: {runs_summary_path}")
    if not os.path.exists(logs_dir):
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    # Load the runs summary
    with open(runs_summary_path, 'r') as f:
        runs_summary = json.load(f)

    enhanced_summary = []

    # Process each run in the summary
    for run in runs_summary:
        run_id = run.get("run_id")
        log_file = run.get("log_file")

        if not run_id or not log_file:
            print(f"Skipping invalid run entry: {run}")
            enhanced_summary.append(run)  # Keep the original entry as-is
            continue

        # Normalize path separators and extract the filename
        if not isinstance(log_file, str) or not log_file.strip():
            print(f"Invalid log_file for run {run_id}: {log_file}")
            enhanced_summary.append(run)  # Keep the original entry as-is
            continue

        log_file_normalized = log_file.replace("\\", "/")  # Replace backslashes with forward slashes
        log_filename = os.path.basename(log_file_normalized)  # Extract the filename
        log_path = os.path.join(logs_dir, log_filename)  # Construct the full path

        if not os.path.exists(log_path):
            print(f"Log file not found for run {run_id}: {log_path}")
            enhanced_summary.append(run)  # Keep the original entry as is
            continue

        # Extract the best individual from the log file
        try:
            best_individual, log_full_path = extract_best_individual(log_path)
            enhanced_entry = run.copy()
            enhanced_entry["best_individual"] = best_individual
            enhanced_entry["log_file"] = log_filename
            enhanced_summary.append(enhanced_entry)
            print(f"Best individual extracted from log: {log_full_path}")
        except Exception as e:
            print(f"Failed to process log for run {run_id}: {e}")
            enhanced_summary.append(run)  # Keep the original entry as-is

    return enhanced_summary