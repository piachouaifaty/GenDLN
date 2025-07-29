import os
import json
from genetic_dln.src.log_processor.log_processor import extract_best_from_run_summary
from genetic_dln.src.validation.validator import validate_individual


def validate_run_summary(run_summary_path, logs_dir, output_dir):
    """
    Enhances a run summary by adding validation results for the best individual of each run.
    Saves progress after each validated run.

    Args:
        run_summary_path (str): Path to the run summary file.
        logs_dir (str): Directory containing log files referenced in the run summary.
        output_dir (str): Directory where the validation summary should be saved.

    Returns:
        dict: Validation summary containing:
            - 'runs': List of validated run entries (each containing original fields + best individual (incl metrics)
              + 'validation_results').
            - 'original_summary_file': Filename of the original summary.
    """
    if not os.path.exists(run_summary_path):
        raise FileNotFoundError(f"Run summary file not found: {run_summary_path}")
    if not os.path.exists(logs_dir):
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Extract the filename of the original run summary
    original_summary_filename = os.path.basename(run_summary_path)

    # Enhance the run summary with the best individuals
    enhanced_run_summary = extract_best_from_run_summary(run_summary_path, logs_dir)

    validated_runs = []
    validation_summary_path = os.path.join(output_dir,
                                           os.path.splitext(original_summary_filename)[0] + "_validation.json")

    # Iterate through each run in the enhanced summary
    for run in enhanced_run_summary:
        print("--------------------")
        run_id = run.get("run_id")
        print(f"Starting validation for {run_id}")

        best_individual = run.get("best_individual")
        if not best_individual:
            print(f"Skipping run {run_id} as no best individual was found.")
            validated_entry = run.copy()
            validated_entry["validation_results"] = "ERROR: Validation skipped due to absence of best individual."
            validated_runs.append(validated_entry)
        else:
            individual = best_individual.get("individual")
            prompt_1 = individual.get("prompt_1")
            prompt_2 = individual.get("prompt_2")
            multi_label = run.get("config", {}).get("multi_label", False)

            if not prompt_1 or not prompt_2:
                print(f"Skipping run {run_id} due to missing prompts.")
                validated_entry = run.copy()
                validated_entry["validation_results"] = "ERROR: Validation skipped due to presence of empty prompt."
                validated_runs.append(validated_entry)
            else:
                try:
                    validation_results = validate_individual(prompt_1, prompt_2, multi_label=multi_label)
                    validated_entry = run.copy()
                    validated_entry["validation_results"] = validation_results
                    validated_runs.append(validated_entry)
                    print(f"Validation results for run {run_id}")
                    print(validation_results)
                except Exception as e:
                    print(f"Validation failed for run {run_id}: {e}")
                    validated_runs.append(run)

        print("--------------------")

        validation_summary = {"runs": validated_runs, "original_summary_file": original_summary_filename}
        with open(validation_summary_path, "w") as f:
            json.dump(validation_summary, f, indent=4)

        print(f"Progress saved: {validation_summary_path}")

    return validation_summary
