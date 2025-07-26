library(data.table)

library(data.table)
library(jsonlite)
library(dplyr)
library(stringr)

build_all_run_summaries <- function(summary_list_dir, logs_dir, validation_dir, output_csv) {
  
  # Ensure the multi_run_batch_summaries.csv exists
  if (!file.exists(summary_list_dir)) {
    stop("The multi_run_batch_summaries.csv does not exist.")
  }
  
  summaries_data <- fread(summary_list_dir)
  print("Loaded multi_run_batch_summaries.csv:")
  print(head(summaries_data))
  
  # Prepare a list for new entries
  new_entries <- list()
  
  for (i in seq_len(nrow(summaries_data))) {
    summary_row <- summaries_data[i, ]
    print(paste0("Processing summary row ", i, ":"))
    
    log_files <- strsplit(summary_row$log_files, ",\\s*")[[1]]
    
    for (log_file in log_files) {
      ga_log_filename <- gsub("^logs/", "", log_file)
      ga_log_filename <- gsub("\\\\", "/", ga_log_filename)
      log_path <- file.path(logs_dir, ga_log_filename)
      log_path <- normalizePath(log_path, winslash = "/", mustWork = FALSE)
      
      print(paste0("Processing log file: ", log_path))
      
      if (!file.exists(log_path)) {
        warning(paste("Log file not found:", log_path))
        next
      }
      
      log_info_flat <- extract_log_info(log_path, flatten = TRUE)
      
      log_info_flat$summary_filename <- summary_row$summary_filename
      validation_filename <- gsub("\\.json$", "_validation.json", summary_row$summary_filename)
      validation_path <- file.path(validation_dir, validation_filename)
      validation_path <- normalizePath(validation_path, winslash = "/", mustWork = FALSE)
      
      # Handle validation_filename consistently
      log_info_flat$validation_filename <- ifelse(file.exists(validation_path), validation_filename, NA_character_)
      
      # Set default values for validation metrics
      log_info_flat$validated <- FALSE
      log_info_flat$val_accuracy <- 0
      log_info_flat$val_macro_avg_f1_score <- 0
      log_info_flat$val_weighted_avg_f1_score <- 0
      log_info_flat$validation_results <- NA_character_ # Using NA_character_
      
      # Check if validation file exists and only then process it
      if (file.exists(validation_path)) {
        validation_content <- fromJSON(validation_path)
        validation_info <- validation_content$runs %>% 
          filter(log_file == ga_log_filename)
        
        if (nrow(validation_info) == 1) {
          extracted_info <- extract_validation_entry_info(validation_info[1, ])
          log_info_flat$validated <- TRUE
          log_info_flat$validation_results <- extracted_info$validation_results
        }
      }
      
      new_entries <- append(new_entries, list(log_info_flat))
    }
  }
  
  # Combine new entries into a data frame using explicit type handling
  if (length(new_entries) > 0) {
    # Identify and standardize column types
    all_columns <- unique(unlist(lapply(new_entries, names)))
    
    # Initialize an empty list to store standardized data
    standardized_entries <- list()
    
    entry_counter <- 0
    first_entry_classes <- NULL
    
    for (entry in new_entries) {
      entry_counter <- entry_counter + 1
      print(paste("Processing entry:", entry_counter))
      
      # Create a new entry with consistent columns
      standardized_entry <- lapply(setNames(all_columns, all_columns), function(col) {
        if (col %in% names(entry)) {
          val <- if (is.list(entry[[col]]) && length(entry[[col]]) == 1) {
            entry[[col]][[1]]
          } else {
            entry[[col]]
          }
          
          # Convert run_start_time to character
          if (col == "run_start_time") {
            return(as.character(val))
          }
          
          # Explicitly handle validation_results as character
          if (col == "validation_results") {
            return(as.character(val))
          }
          
          if (is.null(val)) {
            return(NA_character_)
          } else if (is.na(val)) {
            if (is.numeric(entry[[col]])) {
              return(NA_real_)
            } else if (is.logical(entry[[col]])) {
              return(NA)
            } else {
              return(NA_character_)
            }
          } else {
            return(val)
          }
        } else {
          return(NA_character_)
        }
      })
      
      # Capture classes of the first entry
      if (entry_counter == 1) {
        first_entry_classes <- sapply(standardized_entry, class)
      }
      
      # Check for class mismatch relative to the first entry
      if (entry_counter > 1) {
        for (col_index in seq_along(standardized_entry)) {
          if (length(standardized_entry[[col_index]]) > 0 && length(first_entry_classes[[col_index]]) > 0) {
            if (class(standardized_entry[[col_index]]) != first_entry_classes[col_index]) {
              print(paste("Class mismatch detected at entry", entry_counter, "column", all_columns[col_index], 
                          "Current Class:", class(standardized_entry[[col_index]]),
                          "Expected Class:", first_entry_classes[col_index]))
              print(paste("Value:", standardized_entry[[col_index]]))
              
              # Print the problematic entry for inspection
              cat("Problematic entry:\n")
              print(entry)
            }
          }
        }
      }
      
      # Convert to data.frame and add to standardized_entries
      standardized_entries <- append(standardized_entries, list(as.data.frame(standardized_entry, stringsAsFactors = FALSE)))
    }
    
    # Handle case where first entry might be empty
    if (length(standardized_entries) == 0) {
      stop("No valid entries found to process.")
    }
    
    # Use rbindlist with fill = TRUE
    combined_data <- rbindlist(standardized_entries, fill = TRUE, use.names = TRUE)
    
    # Additional coercion after rbindlist (if necessary)
    for (col in c("runtime_mins", "initial_gen_best_fitness", "initial_gen_best_accuracy", "best_fitness", "best_accuracy")) {
      if (col %in% names(combined_data)) {
        combined_data[, (col) := as.numeric(get(col))]
      }
    }
    
    print("Combined data:")
    print(head(combined_data))
    
    fwrite(combined_data, output_csv)
    print(paste("Output file", output_csv, "overwritten successfully."))
  } else {
    print("No new GA logs to process.")
  }
  
  return(combined_data)
}



process_run_summaries <- function(all_run_summaries_csv) {
  # Read the input CSV
  all_run_summaries <- fread(all_run_summaries_csv)
  print("Loaded all run summaries:")
  print(head(all_run_summaries))
  
  # Check if the required columns exist
  if (!"task" %in% names(all_run_summaries) || !"raw_metrics" %in% names(all_run_summaries) || !"validation_results" %in% names(all_run_summaries)) {
    stop("The required columns 'task', 'raw_metrics', or 'validation_results' are missing.")
  }
  
  # Function to clean and extract metric from text
  clean_and_extract_metric <- function(pattern, text) {
    if (is.na(text)) return(NA)
    # Clean up double quotes
    text <- gsub('""', '"', text)
    value <- regmatches(text, regexec(pattern, text))[[1]]
    if (length(value) > 1) {
      return(as.numeric(value[2]))
    }
    return(NA)
  }
  
  # Add extracted metrics to binary dataframe
  process_binary <- function(df) {
    df[, `:=`(
      best_macro_avg_f1_score = sapply(raw_metrics, function(x) clean_and_extract_metric('"macro avg_f1-score":([0-9.]+)', x)),
      best_weighted_avg_f1_score = sapply(raw_metrics, function(x) clean_and_extract_metric('"weighted avg_f1-score":([0-9.]+)', x)),
      val_macro_avg_f1_score = ifelse(validated, sapply(validation_results, function(x) {
        if (grepl("ERROR", x)) return(NA)
        clean_and_extract_metric('"macro avg_f1-score":([0-9.]+)', x)
      }), 0), # Set to 0 if not validated
      val_weighted_avg_f1_score = ifelse(validated, sapply(validation_results, function(x) {
        if (grepl("ERROR", x)) return(NA)
        clean_and_extract_metric('"weighted avg_f1-score":([0-9.]+)', x)
      }), 0), # Set to 0 if not validated
      val_accuracy = ifelse(validated, sapply(validation_results, function(x) {
        if (grepl("ERROR", x)) return(NA)
        clean_and_extract_metric('"accuracy":([0-9.]+)', x)
      }), 0)  # Set to 0 if not validated
    )]
    return(df)
  }
  
  # Add extracted metrics to multiclass dataframe
  process_multiclass <- function(df) {
    df[, `:=`(
      best_macro_avg_f1_score = sapply(raw_metrics, function(x) clean_and_extract_metric('"macro_avg":\\{"precision":[0-9.]+,"recall":[0-9.]+,"f1_score":([0-9.]+)', x)),
      best_weighted_avg_f1_score = sapply(raw_metrics, function(x) clean_and_extract_metric('"weighted_avg":\\{"precision":[0-9.]+,"recall":[0-9.]+,"f1_score":([0-9.]+)', x)),
      val_macro_avg_f1_score = ifelse(validated, sapply(validation_results, function(x) {
        if (grepl("ERROR", x)) return(NA)
        clean_and_extract_metric('"macro_avg":\\{"precision":[0-9.]+,"recall":[0-9.]+,"f1_score":([0-9.]+)', x)
      }), 0), # Set to 0 if not validated
      val_weighted_avg_f1_score = ifelse(validated, sapply(validation_results, function(x) {
        if (grepl("ERROR", x)) return(NA)
        clean_and_extract_metric('"weighted_avg":\\{"precision":[0-9.]+,"recall":[0-9.]+,"f1_score":([0-9.]+)', x)
      }), 0), # Set to 0 if not validated
      val_accuracy = ifelse(validated, sapply(validation_results, function(x) {
        if (grepl("ERROR", x)) return(NA)
        clean_and_extract_metric('"accuracy":([0-9.]+)', x)
      }), 0)  # Set to 0 if not validated
    )]
    return(df)
  }
  
  # Split into binary and multiclass
  binary_summaries <- all_run_summaries[task == "binary"]
  multi_summaries <- all_run_summaries[task == "multiclass"]
  
  # Process the dataframes
  if (nrow(binary_summaries) > 0) {
    binary_summaries <- process_binary(binary_summaries)
  }
  
  if (nrow(multi_summaries) > 0) {
    multi_summaries <- process_multiclass(multi_summaries)
  }
  
  col_order = c("ga_log_filename", "run_start_time", "runtime_mins", "task",  "initial_gen_best_fitness",   "initial_gen_best_accuracy",  "prompt_1", "prompt_2", "best_fitness", "best_accuracy", "val_accuracy", "best_macro_avg_f1_score", "val_macro_avg_f1_score", "best_weighted_avg_f1_score", "val_weighted_avg_f1_score", "fitness_function", "elitism", "selection_strategy", "selection_params", "crossover_type","crossover_rate",  "mutation_type",       "mutation_rate","mutate_elites", "population_size", "planned_generations", "completed_generations", "stopped_early", "stopped_early_reason", "early_stop_settings", "raw_metrics", "validation_results", "llm_temperatures", "system_info", "summary_filename", "validation_filename")
  
  setcolorder(binary_summaries, col_order)
  setcolorder(multi_summaries, col_order)
  
  # Return the two dataframes as a list
  return(list(
    binary = binary_summaries,
    multiclass = multi_summaries
  ))
}

extract_validation_entry_info <- function(validation_entry) {
  if (!is.null(validation_entry$validation_results) && length(validation_entry$validation_results) > 0) {
    # Validated entry: Convert to JSON string
    validation_results_json <- toJSON(validation_entry$validation_results, auto_unbox = TRUE, pretty = FALSE)
  } else {
    # Non-validated entry: Return a consistent JSON structure with status and an empty list
    validation_results_json <- toJSON(list(status = "empty", results = list()), auto_unbox = TRUE, pretty = FALSE)
  }
  
  # Return a list with a consistent structure
  return(list(
    validated = !is.null(validation_entry$validation_results) && length(validation_entry$validation_results) > 0,
    validation_results = validation_results_json
  ))
}

