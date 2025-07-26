
# NORMALIZE GA LOG TO FORMAT
normalize_ga_log <- function(file_path) {
  # Load the JSON file
  data <- fromJSON(file_path, flatten = FALSE)
  
  # Check and update "run_config" inside hyperparameters
  if (!identical(data$hyperparameters$run_config, data$config)) {
    data$hyperparameters$run_config <- data$config
  }
  
  # Extract general metadata as a list
  
  normalized_path <- gsub("\\\\", "/", data$ga_log_filename)
  
  metadata <- list(
    early_stopping = data$early_stopping,
    runtime = data$runtime,
    system_info = data$system_info,
    config = data$config,
    hyperparameters = data$hyperparameters,
    ga_log_filename = basename(normalized_path)
  )
  
  # Extract generations data
  gen_data <- data$generations
  
  # Extract the first generation (generation 0)
  initial_generation <- gen_data[1, ] # Extract the first row as a data frame
  
  # Extract the remaining generations
  run_generations <- gen_data[-1, ] # Exclude the first row
  run_generations <- run_generations # Convert it to a list if needed
  
  # Count the number of run generations
  num_run_generations <- length(run_generations$generation)
  
  # Return results
  list(
    initial_generation = initial_generation,
    run_generations = run_generations,
    completed_generations = num_run_generations,
    metadata = metadata)
}


## EXTRACT LOG INFO ###
extract_log_info <- function(file_path, flatten = FALSE) {
  
  # This function processes a GA log JSON file to extract metadata, 
  # parameters, and results in a structured format.
  # 
  # Input: file_path (string) - Path to the GA log JSON file.
  # Output: A list containing:
  # - info: General metadata such as filename, system info, runtime, etc.
  # - ga_params: GA parameters including task type, fitness function, 
  #              population size, and strategy details.
  # - results: Results from the initial and last generation.
  
  
  # Normalize the GA log
  log_data <- normalize_ga_log(file_path)
  
  # Extract metadata
  metadata <- log_data$metadata
  
  #extract relevant fields from metadata
  
  ga_log_filename = metadata$ga_log_filename
  
  task_fitness <- extract_task_and_fitness(metadata)
  
  task <- task_fitness$task
  fitness_function <- task_fitness$fitness_function
  
  run_start_time = as.POSIXct(format(as.POSIXct(gsub("ga_log_|\\.log", "", ga_log_filename), format = "%Y%m%d_%H%M%S", tz = "CET")))
  runtime_mins = as.numeric(sub(" .*", "", metadata$runtime))
  
  population_size = metadata$config$population_size
  planned_generations = metadata$config$generations
  completed_generations = log_data$completed_generations
  stopped_early = metadata$early_stopping$status
  stopped_early_reason = metadata$early_stopping$reason
  
  selection_strategy = metadata$config$selection_strategy
  # Conditionally construct selection_params
  selection_params <- if (selection_strategy == "tournament") {
    paste(paste(names(metadata$config$selection_params), 
                unlist(metadata$config$selection_params), 
                sep = ": "), 
          collapse = ", ")
  } else {
    "" # Empty string if selection_strategy is not "tournament"
  }
  
  elitism = metadata$config$elitism_k
  
  crossover_type = metadata$config$crossover_type
  crossover_rate = metadata$config$crossover_rate
  
  mutation_type = metadata$config$mutation_type
  mutation_rate = metadata$config$mutation_rate
  mutate_elites = metadata$config$mutate_elites
  
  #llm_temperatures = paste(paste(names(metadata$config$llm_configs), unlist(metadata$config$llm_configs), sep = ": "), collapse = ", ")
  llm_temperatures = jsonlite::toJSON(metadata$config$llm_configs, auto_unbox = TRUE, pretty = FALSE)
  
  #early_stop_settings = paste(paste(names(metadata$config$early_stopping), unlist(metadata$config$early_stopping), sep = ": "), collapse = ", ")
  early_stop_settings = jsonlite::toJSON(metadata$config$early_stopping, auto_unbox = TRUE, pretty = FALSE)
  
  #system_info = paste(paste(names(metadata$system_info), unlist(metadata$system_info), sep = ": "), collapse = ", ")
  system_info = jsonlite::toJSON(metadata$system_info, auto_unbox = TRUE, pretty = FALSE)
  
  # Extract fitness information from initial generation (generation 0)
  gen_0 = log_data$initial_generation
  initial_gen_best_fitness = gen_0$fitness_data$best_individual$fitness_score
  initial_gen_best_accuracy = gen_0$fitness_data$best_individual$raw_metrics$accuracy
  
  # LAST GENERATION (RESULTS)
  last_gen = tail(log_data$run_generations, n = 1)
  
  best = last_gen$fitness_data$best_individual
  prompt_1 = best$individual$prompt_1
  prompt_2 = best$individual$prompt_2
  
  
  best_fitness = best$fitness_score
  best_accuracy = best$raw_metrics$accuracy
  #raw_metrics = paste(paste(names(best$raw_metrics), unlist(best$raw_metrics), sep = ": "), collapse = ", ")
  raw_metrics = jsonlite::toJSON(best$raw_metrics, auto_unbox = TRUE, pretty = FALSE)
  
  
  #----- FORMAT OF NORMALIZED GA LOG ------#
    #normalizedlog$info
  info <- list(
    ga_log_filename = ga_log_filename,
    system_info = system_info,
    run_start_time = run_start_time,
    runtime_mins = runtime_mins,
    task = task
  )
  #normalizedlog$ga_params
  ga_params <- list(
    fitness_function = fitness_function,
    population_size = population_size,
    planned_generations = planned_generations,
    completed_generations = completed_generations,
    early_stop_settings = early_stop_settings,
    stopped_early = stopped_early,
    stopped_early_reason = stopped_early_reason,
    selection_strategy = selection_strategy,
    selection_params = selection_params,
    elitism = elitism,
    crossover_type = crossover_type,
    crossover_rate = crossover_rate,
    mutation_type = mutation_type,
    mutation_rate = mutation_rate,
    mutate_elites = mutate_elites,
    llm_temperatures = llm_temperatures
  )
  
  ##normalizedlog$results
  results <- list(
    initial_gen_best_fitness = initial_gen_best_fitness,
    initial_gen_best_accuracy = initial_gen_best_accuracy,
    prompt_1 = prompt_1,
    prompt_2 = prompt_2,
    best_fitness = best_fitness,
    best_accuracy = best_accuracy,
    raw_metrics = raw_metrics
  )
  
  unflattened_output = list(
    info = info,
    ga_params = ga_params,
    results = results)
  
  #helper function for alternate output
  flatten_log_results <- function(extracted_data) {
    # Combine all lists into a single flat dataframe
    cbind(
      as.data.frame(lapply(extracted_data$info, unlist)),
      as.data.frame(lapply(extracted_data$ga_params, unlist)),
      as.data.frame(lapply(extracted_data$results, unlist))
    )
  }
  
  if (flatten)
  {
   flattened_output = flatten_log_results(unflattened_output)
   flattened_output
  }
  else {unflattened_output}
  
}


# EXTRACT TASK AND FITNESS FROM GA_LOG METADATA (internal function)
extract_task_and_fitness <- function(metadata) {
  # classification type: binary or multi
  task <- if (metadata$config$multi_label) {
    "multiclass"
  } else {
    "binary"
  }
  
  # Initialize fitness_function as an empty string
  fitness_function <- ""
  
  if (task == "binary") {
    # Binary case: Extract non-zero weights from binary_weights
    binary_weights <- metadata$hyperparameters$binary_weights
    non_zero_weights <- binary_weights[binary_weights != 0]
    
    # Build the fitness function string
    fitness_function <- paste(
      sapply(names(non_zero_weights), function(name) {
        paste0(non_zero_weights[[name]], "*", name)
      }),
      collapse = " + "
    )
    
  } else if (task == "multiclass") {
    # Multiclass case: Extract non-zero weights from multi_label_weights
    multi_label_weights <- metadata$hyperparameters$multi_label_weights
    
    # Extract and check top-level accuracy
    fitness_components <- c()
    if (multi_label_weights$accuracy != 0) {
      fitness_components <- c(
        fitness_components,
        paste0(multi_label_weights$accuracy, "*accuracy")
      )
    }
    
    # Extract and check macro_avg non-zero components
    macro_avg <- multi_label_weights$macro_avg
    macro_non_zero <- macro_avg[macro_avg != 0]
    fitness_components <- c(
      fitness_components,
      sapply(names(macro_non_zero), function(name) {
        paste0(macro_non_zero[[name]], "*macro_avg_", name)
      })
    )
    
    # Extract and check weighted_avg non-zero components
    weighted_avg <- multi_label_weights$weighted_avg
    weighted_non_zero <- weighted_avg[weighted_avg != 0]
    fitness_components <- c(
      fitness_components,
      sapply(names(weighted_non_zero), function(name) {
        paste0(weighted_non_zero[[name]], "*weighted_avg_", name)
      })
    )
    
    # Combine all components into the fitness function
    fitness_function <- paste(fitness_components, collapse = " + ")
  }
  
  # Return both task and fitness function
  list(task = task, fitness_function = fitness_function)
}





####HERE LOG_DATA REFERS TO NORMALIZED LOG DATA 
# Extract binary metrics for a specific generation
parse_binary_metrics <- function(log_data, generation) {
  # Access the specific generation's entry in fitness_summary
  fitness_entry <- log_data$run_generations$fitness_data
  
  if (is.null(fitness_entry)) {
    stop(paste("Generation", generation, "not found in binary log data."))
  }
  
  # Access the best individual and its metrics for the given generation
  best_individual_fitness <- fitness_entry$best_individual$fitness_score[[generation]]
  average_fitness <- fitness_entry$average_fitness[[generation]]
  raw_metrics = fitness_entry$best_individual$raw_metrics
  
  # Extract key metrics
  metrics <- list(
    generation = generation,
    average_fitness = average_fitness,
    fitness = best_individual_fitness,
    accuracy = raw_metrics$accuracy[[generation]],
    macro_avg_f1_score = raw_metrics[["macro avg_f1-score"]][[generation]],
    weighted_avg_f1_score = raw_metrics[["weighted avg_f1-score"]][[generation]]
  )
  
  return(metrics)
}

# Extract multiclass metrics for a specific generation
parse_multiclass_metrics <- function(log_data, generation) {
  # Access the specific generation's entry in fitness_summary
  fitness_entry <- log_data$run_generations$fitness_data
  
  if (is.null(fitness_entry)) {
    stop(paste("Generation", generation, "not found in multiclass log data."))
  }
  
  # Access the best individual and its metrics for the given generation
  best_individual_fitness <- fitness_entry$best_individual$fitness_score[[generation]]
  average_fitness <- fitness_entry$average_fitness[[generation]]
  raw_metrics = fitness_entry$best_individual$raw_metrics
  
  
  # Extract key metrics
  metrics <- list(
    generation = generation,
    average_fitness = average_fitness,
    fitness = best_individual_fitness,
    accuracy = raw_metrics$accuracy[[generation]],
    macro_avg_precision = raw_metrics$macro_avg$precision[[generation]],
    macro_avg_recall = raw_metrics$macro_avg$recall[[generation]],
    macro_avg_f1_score = raw_metrics$macro_avg$f1_score[[generation]],
    weighted_avg_precision = raw_metrics$weighted_avg$precision[[generation]],
    weighted_avg_recall = raw_metrics$weighted_avg$recall[[generation]],
    weighted_avg_f1_score = raw_metrics$weighted_avg$f1_score[[generation]]
  )
  
  return(metrics)
}

# General function to extract metrics based on task and generation
extract_ga_metrics <- function(log_data, generation) {
  task <- ifelse(log_data$metadata$config$multi_label, "multiclass", "binary")
  
  if (task == "binary") {
    return(parse_binary_metrics(log_data, generation))
  } else if (task == "multiclass") {
    return(parse_multiclass_metrics(log_data, generation))
  } else {
    stop("Unknown task type.")
  }
}

#normalized
############################




parse_filename_to_date <- function(filename, prefix = "ga_log_", suffix = ".log", format = "%Y%m%d_%H%M%S") {
  # Remove prefix and suffix
  cleaned_filename <- gsub(paste0("^", prefix), "", filename)
  cleaned_filename <- gsub(paste0(suffix, "$"), "", cleaned_filename)
  
  # Parse the cleaned filename to date-time
  parsed_date <- as.POSIXct(cleaned_filename, format = format, tz = "CET")
  
  if (is.na(parsed_date)) {
    stop("Failed to parse date from filename. Ensure the filename matches the expected format.")
  }
  
  return(parsed_date)
}




generate_ga_report <- function(file_path) {
  # Normalize the GA log
  log_data <- normalize_ga_log(file_path)
  log_info <- extract_log_info(file_path)
  
  # Extract general configuration details
  metadata <- log_data$metadata
  config <- metadata$config
  task_fitness <- extract_task_and_fitness(metadata)
  task <- task_fitness$task
  fitness_function <- task_fitness$fitness_function
  
  
  # general configs (stuff that can be extracted from log_info/log_data
  general_config <- list(
    log_filename = log_data[["metadata"]][["ga_log_filename"]],
    run_date = parse_filename_to_date(basename(file_path)),
    runtime = log_data[["metadata"]][["runtime"]],
    task = if (log_data[["metadata"]][["config"]][["multi_label"]]) {"Multi-Label Classification"} else{"Binary Classification"},
    
    # Results of the initial generation (gen 0)
    initial_gen_best_fitness <- log_info$results$initial_gen_best_fitness,
    initial_gen_best_accuracy <- log_info$results$initial_gen_best_accuracy,
    
    # Results of the last generation
    best_fitness = log_info$results$best_fitness,
    best_accuracy = log_info$results$best_accuracy,
    raw_metrics = log_info$results$raw_metrics,
    prompt_1 = as.character(log_info$results$prompt_1),
    prompt_2 = as.character(log_info$results$prompt_2),
    
    
    fitness_function = log_info$ga_params$fitness_function,
    elitism = log_info$ga_params$elitism,
    population_size = log_info$ga_params$population_size,
    planned_generations = log_info$ga_params$planned_generations,
    completed_generations = log_info$ga_params$completed_generations,
    selection_strategy = log_info$ga_params$selection_strategy,
    selection_params = log_info$ga_params$selection_params,
    crossover_type = log_info$ga_params$crossover_type,
    crossover_rate = log_info$ga_params$crossover_rate,
    mutation_type = log_info$ga_params$mutation_type,
    mutation_rate = log_info$ga_params$mutation_rate,
    mutate_elites = log_info$ga_params$mutate_elites,
    early_stop_settings = log_info$ga_params$early_stop_settings,
    stopped_early = log_info$ga_params$stopped_early,
    stopped_early_reason = log_info$ga_params$stopped_early_reason,
    llm_temperatures = log_info$ga_params$llm_temperatures
  )
  
  
  
  best_individual = log_data$run_generations$fitness_data$best_individual
  worst_individual = log_data$run_generations$fitness_data$worst_individual
  avg_fitness = log_data$run_generations$fitness_data$average_fitness
  
  metrics <- list(
    accuracy = numeric(),
    fitness_score = numeric(),
    average_fitness = numeric()
  )
  
  # Extract metrics for each generation
  ga_journey <- lapply(seq_len(general_config$completed_generations), function(gen) {
    if (task == "binary") {
      list(
        generation = gen,
        best = list(
          accuracy = best_individual$raw_metrics$accuracy[[gen]],
          fitness_score = best_individual$fitness_score[[gen]],
          macro_avg_f1_score = best_individual$raw_metrics[["macro avg_f1-score"]][[gen]],
          weighted_avg_f1_score = best_individual$raw_metrics[["weighted avg_f1-score"]][[gen]]
        ),
        worst = list(
          accuracy = worst_individual$raw_metrics$accuracy[[gen]],
          fitness_score = worst_individual$fitness_score[[gen]],
          macro_avg_f1_score = worst_individual$raw_metrics[["macro avg_f1-score"]][[gen]],
          weighted_avg_f1_score = worst_individual$raw_metrics[["weighted avg_f1-score"]][[gen]]
        ),
        average_fitness = avg_fitness[[gen]]
      )
    } else if (task == "multiclass") {
      list(
        generation = gen,
        best = list(
          accuracy = best_individual$raw_metrics$accuracy[[gen]],
          fitness_score = best_individual$fitness_score[[gen]],
          macro_avg_f1_score = best_individual$raw_metrics$macro_avg$f1_score[[gen]],
          weighted_avg_f1_score = best_individual$raw_metrics$weighted_avg$f1_score[[gen]]
        ),
        worst = list(
          accuracy = worst_individual$raw_metrics$accuracy[[gen]],
          fitness_score = worst_individual$fitness_score[[gen]],
          macro_avg_f1_score = worst_individual$raw_metrics$macro_avg$f1_score[[gen]],
          weighted_avg_f1_score = worst_individual$raw_metrics$weighted_avg$f1_score[[gen]]
        ),
        average_fitness = avg_fitness[[gen]]
      )
    }
  })
  
  # Extract numerical metrics for statistical summary
  accuracy_vals <- unlist(lapply(ga_journey, function(gen) gen$best$accuracy))
  fitness_vals <- unlist(lapply(ga_journey, function(gen) gen$best$fitness_score))
  avg_fitness_vals <- unlist(lapply(ga_journey, function(gen) gen$average_fitness))
  macro_avg_f1_vals <- unlist(lapply(ga_journey, function(gen) gen$best$macro_avg_f1_score))
  weighted_avg_f1_vals <- unlist(lapply(ga_journey, function(gen) gen$best$weighted_avg_f1_score))
  
  # Statistical summary
  calculate_stats <- function(values) {
    list(
      mean = mean(values, na.rm = TRUE),
      sd = sd(values, na.rm = TRUE),
      min = min(values, na.rm = TRUE),
      max = max(values, na.rm = TRUE)
    )
  }
  
  accuracy_stats <- calculate_stats(accuracy_vals)
  fitness_stats <- calculate_stats(fitness_vals)
  avg_fitness_stats <- calculate_stats(avg_fitness_vals)
  macro_avg_f1_stats <- calculate_stats(macro_avg_f1_vals)
  weighted_avg_f1_stats <- calculate_stats(weighted_avg_f1_vals)
  
  summary_stats <- list(
    Accuracy = accuracy_stats,
    Fitness_Score = fitness_stats,
    Average_Fitness = avg_fitness_stats,
    Macro_Avg_F1 = macro_avg_f1_stats,
    Weighted_Avg_F1 = weighted_avg_f1_stats
  )
  
  # Format summary statistics into a readable string
  summary_text <- paste(
    "===== Summary Statistics Across Generations =====",
    paste("Metric: Accuracy",
          paste0("  Mean: ", round(accuracy_stats$mean, 4)),
          paste0("  SD  : ", round(accuracy_stats$sd, 4)),
          paste0("  Min : ", round(accuracy_stats$min, 4)),
          paste0("  Max : ", round(accuracy_stats$max, 4)),
          sep = "\n"),
    paste("Metric: Fitness Score",
          paste0("  Mean: ", round(fitness_stats$mean, 4)),
          paste0("  SD  : ", round(fitness_stats$sd, 4)),
          paste0("  Min : ", round(fitness_stats$min, 4)),
          paste0("  Max : ", round(fitness_stats$max, 4)),
          sep = "\n"),
    paste("Metric: Average Fitness",
          paste0("  Mean: ", round(avg_fitness_stats$mean, 4)),
          paste0("  SD  : ", round(avg_fitness_stats$sd, 4)),
          paste0("  Min : ", round(avg_fitness_stats$min, 4)),
          paste0("  Max : ", round(avg_fitness_stats$max, 4)),
          sep = "\n"),
    paste("Metric: Macro Avg F1",
          paste0("  Mean: ", round(macro_avg_f1_stats$mean, 4)),
          paste0("  SD  : ", round(macro_avg_f1_stats$sd, 4)),
          paste0("  Min : ", round(macro_avg_f1_stats$min, 4)),
          paste0("  Max : ", round(macro_avg_f1_stats$max, 4)),
          sep = "\n"),
    paste("Metric: Weighted Avg F1",
          paste0("  Mean: ", round(weighted_avg_f1_stats$mean, 4)),
          paste0("  SD  : ", round(weighted_avg_f1_stats$sd, 4)),
          paste0("  Min : ", round(weighted_avg_f1_stats$min, 4)),
          paste0("  Max : ", round(weighted_avg_f1_stats$max, 4)),
          sep = "\n"),
    sep = "\n\n"
  )
  
  
  
  
  # Format the report
  report <- paste(
    "=====================",
    "Genetic Algorithm Report",
    "=====================",
    "\nGeneral Configuration:",
    paste0("  Log Filename       : ", general_config$log_filename),
    paste0("  Run Date           : ", general_config$run_date),
    paste0("  Runtime            : ", general_config$runtime),
    paste0("  Classification Task: ", general_config$task),
    paste0("  Fitness Function   : ", general_config$fitness_function),
    paste0("  Elitism            : ", general_config$elitism),
    paste0("  Population Size    : ", general_config$population_size),
    paste0("  Planned Generations: ", general_config$planned_generations),
    paste0("  Completed Generations: ", general_config$completed_generations),
    paste0("  Selection Strategy : ", general_config$selection_strategy),
    paste0("    Selection Parameters: ", general_config$selection_params),
    paste0("  Crossover Type     : ", general_config$crossover_type),
    paste0("  Crossover Rate     : ", general_config$crossover_rate),
    paste0("  Mutation Type      : ", general_config$mutation_type),
    paste0("  Mutation Rate      : ", general_config$mutation_rate),
    paste0("  Mutate Elites      : ", general_config$mutate_elites),
    paste0("  Early Stop Settings: ", general_config$early_stop_settings),
    paste0("  Early Stopping     : ", general_config$stopped_early),
    paste0("  Early Stopping Reason: ", general_config$stopped_early_reason),
    paste0("  LLM Temperatures   : ", general_config$llm_temperatures),
    "\n===== GA Journey =====",
    paste(sapply(ga_journey, function(gen) {
      paste(
        paste0("\nGeneration ", gen$generation, ":", "\n"),
        "\tBest Individual:\n",
        paste0("\t\tAccuracy       : ", gen$best$accuracy, "\n"),
        paste0("\t\tFitness Score  : ", gen$best$fitness_score, "\n"),
        paste0("\t\tMacro Avg F1   : ", gen$best$macro_avg_f1_score, "\n"),
        paste0("\t\tWeighted Avg F1: ", gen$best$weighted_avg_f1_score, "\n"),
        "\tWorst Individual:\n",
        paste0("\t\tAccuracy       : ", gen$worst$accuracy, "\n"),
        paste0("\t\tFitness Score  : ", gen$worst$fitness_score, "\n"),
        paste0("\t\tMacro Avg F1   : ", gen$worst$macro_avg_f1_score, "\n"),
        paste0("\t\tWeighted Avg F1: ", gen$worst$weighted_avg_f1_score, "\n"),
        paste0("\tAverage Fitness      : ", gen$average_fitness, "\n"),
        "\n----------------------------------------"
      )
    }), collapse = "\n"),
    "\nBest Overall Individual:",
    paste0("  Prompt 1: ", general_config$prompt_1),
    paste0("  Prompt 2: ", general_config$prompt_2),
    "\n===== End of GA Journey =====","\n",
    summary_text,
    sep = "\n"
  )
  
  # Print the report
  cat(report)
  
  # Return the report for further use
  return(list(report = report, ga_journey = ga_journey, summary_statistics = summary_stats))
}




