reconstruct_summary_for_partial_run <- function(partial_logs_dir, output_file) {
  # Helper function to parse filenames into dates
  parse_filename_to_date <- function(filename) {
    str_extract(filename, "\\d{8}_\\d{6}") %>% as.POSIXct(format = "%Y%m%d_%H%M%S", tz = "UTC")
  }
  
  # Step 1: Read all log filenames from the directory
  log_filenames <- list.files(partial_logs_dir, pattern = "\\.log$", full.names = TRUE)
  sorted_filenames <- log_filenames[order(sapply(basename(log_filenames), parse_filename_to_date))]
  
  # Step 2: Extract configs from each log file and build the summary
  summary <- lapply(seq_along(sorted_filenames), function(i) {
    log_file <- sorted_filenames[i]
    metadata <- normalize_ga_log(log_file)$metadata
    
    # Flatten the config to ensure scalar values and adjust selection_params structure
    flattened_config <- metadata$config %>%
      lapply(function(x) if (length(x) == 1) x[[1]] else x)
    
    # Ensure selection_params is a dictionary-like structure
    if (!is.null(flattened_config$selection_params)) {
      flattened_config$selection_params <- list(tournament_size = flattened_config$selection_params)
    }
    
    list(
      run_id = sprintf("Run_%02d", i),
      config = flattened_config,
      log_file = file.path("logs", basename(log_file)) # Add "logs/" prefix to match Summary 2
    )
  })
  
  # Convert list to JSON, ensuring no unnecessary arrays are included
  json_output <- toJSON(summary, auto_unbox = TRUE, pretty = TRUE)
  
  # Save the JSON to a file
  write(json_output, file = output_file)
  
  cat(sprintf("Summary successfully saved to %s\n", output_file))
}