

#function to build multi_run_batch_summaries.csv in run_stats_and_mapping
build_run_batch_summaries <- function(summaries_dir, validation_dir, output_csv) {
  # Get all summary and validation files
  summary_files <- list.files(summaries_dir, pattern = "*.json", full.names = TRUE)
  validation_files <- list.files(validation_dir, pattern = "*.json", full.names = TRUE) %>% basename()
  
  # Ensure summary_data exists as an empty tibble if not initialized
  if (!exists("summary_data")) {
    summary_data <- tibble(
      summary_filename = character(),
      num_runs = integer(),
      run_date = as.POSIXct(character()),
      validated = logical(),
      validation_filename = character(),
      log_files = character(),
      Notes = character()
    )
  }
  
  # Iterate over summary files to extract information
  for (file_path in summary_files) {
    # Extract filename (without full path)
    summary_filename <- basename(file_path)
    
    # Check if this summary already exists in the summary_data
    if (summary_filename %in% summary_data$summary_filename) {
      next  # Skip processing if it already exists
    }
    
    # Parse the run date from the filename
    run_date <- as.POSIXct(
      format(as.POSIXct(gsub("ga_runs_summary_|\\.json", "", summary_filename), 
                        format = "%Y%m%d_%H%M%S", tz = "CET"))
    )
    
    # Read the JSON content as a dataframe
    summary_content <- fromJSON(file_path, flatten = TRUE)
    
    # Count the number of runs
    num_runs <- nrow(summary_content)
    
    # Extract log filenames directly from the column
    log_files <- summary_content$log_file %>% 
      gsub("^logs", "", .) %>%      # Remove the "logs/" prefix
      gsub("\\\\", "", .) %>%      # Normalize backslashes to forward slashes
      gsub("/", "", .) %>%
      paste(collapse = ", ")        # Combine into a single string
    
    # Check if a matching validation file exists
    validation_match <- gsub("\\.json$", "_validation.json", summary_filename)
    validated <- validation_match %in% validation_files
    validation_filename <- if (validated) validation_match else ""
    
    # Add a row to the summary data
    summary_data <- summary_data %>%
      add_row(
        summary_filename = summary_filename,
        num_runs = num_runs,
        run_date = run_date,
        validated = validated,
        validation_filename = validation_filename,
        log_files = log_files,
        Notes = ""  # Empty for now, can be filled later
      )
  }
  
  # Handle unmatched validation files
  unmatched_validations <- setdiff(validation_files, summary_data$validation_filename)
  
  for (validation_file in unmatched_validations) {
    # Extract validation filename
    validation_filename <- basename(validation_file)
    
    # Parse the run date from the validation filename
    run_date <- as.POSIXct(
      format(as.POSIXct(gsub("ga_runs_summary_|_validation\\.json", "", validation_filename), 
                        format = "%Y%m%d_%H%M%S", tz = "CET"))
    )
    
    # Add a row for unmatched validation files
    summary_data <- summary_data %>%
      add_row(
        summary_filename = "",  # No matching summary
        num_runs = NA,          # Number of runs is unknown
        run_date = run_date,
        validated = TRUE,
        validation_filename = validation_filename,
        log_files = "",         # No log files for unmatched validations
        Notes = ""              # Empty for now
      )
  }
  
  # Check if output CSV already exists
  if (file.exists(output_csv)) {
    # Read existing data
    existing_data <- read.csv(output_csv)
    
    # Convert run_date in existing_data to POSIXct
    existing_data$run_date <- as.POSIXct(existing_data$run_date)
    
    # Append only new entries
    new_data <- anti_join(summary_data, existing_data, by = "summary_filename")
    
    combined_data = existing_data
    
    # If new_data is empty, skip appending
    if (nrow(new_data) > 0) {
      combined_data <- bind_rows(existing_data, new_data)
      # Save the combined data back to CSV
      write.csv(combined_data, output_csv, row.names = FALSE)
      combined_data
    } else {
      message("No new summaries to append. Existing CSV is up to date.")
      combined_data
    }
  } else {
    # Save the initial data to CSV
    write.csv(summary_data, output_csv, row.names = FALSE)
    summary_data
  }
  
  
}


# ADD NOTES TO SUMMARY /// CSV with NOTE COLUMN
add_notes_to_summary <- function(csv_path, summary_filename, new_note) {
  # Read the existing CSV
  if (!file.exists(csv_path)) {
    stop("CSV file does not exist.")
  }
  
  summary_data <- read.csv(csv_path)
  
  # Ensure summary_filename exists in the data
  if (!summary_filename %in% summary_data$summary_filename) {
    stop("Summary filename not found in the CSV.")
  }
  
  # Append the new note to the existing note
  summary_data$Notes[summary_data$summary_filename == summary_filename] <- 
    paste(summary_data$Notes[summary_data$summary_filename == summary_filename], new_note, sep = "; ")
  
  # Save the updated CSV
  write.csv(summary_data, csv_path, row.names = FALSE)
  
  message("Note added/appended successfully.")
}



