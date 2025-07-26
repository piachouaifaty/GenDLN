metric_colors <- c(
  "accuracy" = "#0072B2",
  "fitness_score" = "#E69F00",
  "average_fitness" = "#009E73",
  "macro_avg_f1_score" = "#D55E00",
  "weighted_avg_f1_score" = "#CC79A7"
)

extract_all_metrics <- function(log_data) {
  # Determine the task type
  task <- ifelse(log_data$metadata$config$multi_label, "multiclass", "binary")
  
  # Extract fitness data
  fitness_data <- log_data$run_generations$fitness_data
  
  if (task == "binary") {
    # Binary task metrics extraction
    metrics <- tibble(
      generation = fitness_data$generation_id,
      accuracy = map_dbl(seq_along(fitness_data$generation_id), 
                         ~ fitness_data$best_individual$raw_metrics$accuracy[[.]]),
      fitness_score = map_dbl(seq_along(fitness_data$generation_id), 
                              ~ fitness_data$best_individual$fitness_score[[.]]),
      average_fitness = fitness_data$average_fitness,
      macro_avg_f1_score = map_dbl(seq_along(fitness_data$generation_id), 
                                   ~ fitness_data$best_individual$raw_metrics$`macro avg_f1-score`[[.]]),
      weighted_avg_f1_score = map_dbl(seq_along(fitness_data$generation_id), 
                                      ~ fitness_data$best_individual$raw_metrics$`weighted avg_f1-score`[[.]])
    )
  } else if (task == "multiclass") {
    # Placeholder for multiclass task metrics extraction
    metrics <- tibble(
      generation = fitness_data$generation_id,
      accuracy = map_dbl(seq_along(fitness_data$generation_id), 
                         ~ fitness_data$best_individual$raw_metrics$accuracy[[.]]),
      fitness_score = map_dbl(seq_along(fitness_data$generation_id), 
                              ~ fitness_data$best_individual$fitness_score[[.]]),
      average_fitness = fitness_data$average_fitness,
      # Placeholder logic for macro_avg_f1_score and weighted_avg_f1_score
      macro_avg_f1_score = map_dbl(seq_along(fitness_data$generation_id), 
                                   ~ fitness_data$best_individual$raw_metrics$macro_avg$f1_score[[.]]),
      weighted_avg_f1_score = map_dbl(seq_along(fitness_data$generation_id), 
                                      ~ fitness_data$best_individual$raw_metrics$weighted_avg$f1_score[[.]])
    )
  } else {
    stop("Unknown task type")
  }
  
  # Pivot longer for easy plotting
  metrics <- metrics %>%
    pivot_longer(
      cols = -generation,
      names_to = "Metric",
      values_to = "Metric_Value"
    ) %>%
    mutate(task = task)
  
  return(metrics)
}


# Generalized Plot Function with Optional Custom Axis Ranges
plot_metrics <- function(file_path, requested_metrics, x_limits = NULL, y_limits = NULL) {
  # Normalize the log file
  log_data <- normalize_ga_log(file_path)
  log_filename <- log_data$metadata$ga_log_filename  # Extract GA log filename for caption
  
  # Extract all available metrics
  metrics <- extract_all_metrics(log_data)
  
  # Filter metrics based on requested ones
  filtered_metrics <- metrics %>%
    filter(Metric %in% requested_metrics)
  
  # Define the x-axis scale conditionally
  x_scale <- if (!is.null(x_limits)) {
    scale_x_continuous(limits = x_limits, breaks = seq(x_limits[1], x_limits[2], by = 3))
  } else {
    scale_x_continuous(breaks = seq(min(filtered_metrics$generation), max(filtered_metrics$generation), by = 3)) 
  }
  
  # Define the y-axis scale conditionally
  y_scale <- if (!is.null(y_limits)) {
    scale_y_continuous(limits = y_limits, breaks = seq(y_limits[1], y_limits[2], by = 0.05)) 
  } else {
    scale_y_continuous(
      breaks = seq(
        floor(min(filtered_metrics$Metric_Value)), 
        ceiling(max(filtered_metrics$Metric_Value)), 
        by = 0.05  # Increase granularity
      )
    )
  }
  
  # Plot the metrics
  ggplot(filtered_metrics, aes(x = generation, y = Metric_Value, color = Metric)) +
    geom_line(linewidth = 1) +  # Updated to use `linewidth`
    geom_point(size = 2) +
    x_scale +  # Apply the conditional x-axis scale
    y_scale +  # Apply the conditional y-axis scale
    scale_color_manual(values = metric_colors) +
    labs(
      title = "Best Individual Metrics by Generation vs. Average Fitness",
      x = "Generation",
      y = NULL, #remove y axis label
      color = NULL  # Remove "Metric" title from legend
      #caption = paste(log_filename)  # Add GA log filename at the bottom
    ) +
    theme_minimal(base_size = 13) +
    theme(
      text = element_text(family = "Arial", size = 14),  # Bigger text for all
      plot.title = element_text(size = 15, hjust = 0.5),  # centered title
      axis.text.x = element_text(size = 12, hjust = 1),  # Smaller x-axis text
      axis.title.x = element_text(size = 11),  # Change x-axis label size
      
      axis.text.y = element_text(size = 12),  # Bigger Y-axis text
      #axis.title.y = element_text(size = 10),  # Change y-axis label size
      
      legend.text = element_text(size = 8),  # Bigger legend text
      legend.position = "bottom",  # Move legend left 
      legend.direction = "horizontal",  # Align legend horizontally
      legend.justification = "center"
    )
    #theme(
    #  text = element_text(family = "Times New Roman", size = 11),
    #  legend.position = "bottom",  # Move legend to the right
    #  legend.direction = "horizontal",  # Display legend items vertically
    #  legend.justification = "center",
    #  legend.text = element_text(family = "Times New Roman", size = 9),  # Smaller font for legend
    #  axis.text.x = element_text(family = "Times New Roman", angle = 0, hjust = 0.5),  # Keep x-axis labels horizontal
    #  panel.grid.major = element_line(linewidth = 0.5),
    #  panel.grid.minor = element_line(linewidth = 0.25, linetype = "dotted"),
    #  plot.caption = element_text(family = "Times New Roman",size = 8, hjust = 0.5, face = "italic")  # Small italic caption
    
}


compute_generation_stats <- function(ga_journey) {
  # Use indexing to ensure extraction works regardless of data type
  df <- do.call(rbind, lapply(ga_journey, function(gen) {
    data.frame(
      generation = gen[["generation"]],
      best_accuracy = as.numeric(gen[["best"]][["accuracy"]]),
      worst_accuracy = as.numeric(gen[["worst"]][["accuracy"]]),
      average_fitness = as.numeric(gen[["average_fitness"]])
    )
  }))
  
  # Compute summary statistics
  stats <- data.frame(
    Metric = c("Accuracy", "Fitness"),
    Mean = colMeans(df[, -1], na.rm = TRUE),
    SD = apply(df[, -1], 2, sd, na.rm = TRUE),
    Min = apply(df[, -1], 2, min, na.rm = TRUE),
    Max = apply(df[, -1], 2, max, na.rm = TRUE)
  )
  
  return(stats)
}





extract_dynamic_metrics <- function(log_data, metric_list) {
  task <- ifelse(log_data$metadata$config$multi_label, "multiclass", "binary")
  
  fitness_data <- log_data$run_generations$fitness_data
  
  metrics <- fitness_data %>%
    mutate(
      generation = generation_id,
      extracted_metrics = map(
        seq_along(fitness_data$generation_id),
        function(gen_idx) {
          metric_values <- sapply(
            metric_list,
            function(metric) {
              tryCatch(
                eval(parse(text = paste0("fitness_data$", metric, "[[", gen_idx, "]]"))),
                error = function(e) NA
              )
            }
          )
          as.data.frame(t(metric_values), stringsAsFactors = FALSE)
        }
      )
    ) %>%
    unnest_wider(extracted_metrics, names_sep = "_extracted_") %>% # Add prefix to avoid conflicts
    pivot_longer(
      cols = starts_with("Metric_") | ends_with("_extracted_"), # Select prefixed columns
      names_to = "Metric",
      values_to = "Metric_Value"
    ) %>%
    mutate(task = task)
  
  return(metrics)
}



plot_convergence <- function(file_path) {
  # Normalize the log file
  log_data <- normalize_ga_log(file_path)
  log_filename <- log_data$metadata$ga_log_filename  # For caption
  
  # Extract relevant data
  fitness_data <- log_data$run_generations$fitness_data
  generations <- log_data$run_generations$generation
  
  if (is.null(fitness_data) || is.null(generations)) {
    stop("No fitness or generation data found.")
  }
  
  # Build convergence data
  convergence_data <- tibble(
    generation = generations,
    best_fitness = map_dbl(seq_along(generations), function(i) {
      fitness_data$best_individual$fitness_score[[i]]
    }),
    actual_worst_fitness = map_dbl(seq_along(generations), function(i) {
      # Filter out -1 values for actual worst fitness
      worst <- fitness_data$worst_individual$fitness_score[[i]]
      if (worst == -1) NA else worst
    }),
    is_invalid = map_lgl(seq_along(generations), function(i) {
      fitness_data$worst_individual$fitness_score[[i]] == -1
    })
  )
  
  # Set axis limits to maintain 80-20 ratio
  y_min <- min(convergence_data$actual_worst_fitness, na.rm = TRUE)
  y_max <- max(convergence_data$best_fitness, na.rm = TRUE)
  expanded_min <- y_min - (y_max - y_min) * 0.2
  
  # Plot convergence
  ggplot(convergence_data, aes(x = generation)) +
    geom_line(aes(y = best_fitness, color = "Best Fitness"), size = 1.2, linetype = "dashed") +
    geom_line(aes(y = actual_worst_fitness, color = "Worst Fitness"), size = 1.2, linetype = "dotted", na.rm = TRUE) +
    geom_ribbon(aes(ymin = actual_worst_fitness, ymax = best_fitness), fill = "blue", alpha = 0.2) +
    geom_point(data = convergence_data %>% filter(is_invalid),
               aes(y = expanded_min), shape = 4, size = 4, color = "red") +
    geom_point(data = convergence_data %>% filter(!is.na(actual_worst_fitness)),
               aes(y = actual_worst_fitness), shape = 21, size = 3, fill = "red") +
    geom_segment(data = convergence_data %>% filter(!is.na(actual_worst_fitness)),
                 aes(y = actual_worst_fitness, yend = best_fitness, xend = generation),
                 color = "blue", linetype = "dotted", linewidth = 0.5) +
    scale_x_continuous(breaks = seq(min(convergence_data$generation), max(convergence_data$generation), by = 3)) +
    scale_y_continuous(
      limits = c(expanded_min, y_max),
      breaks = seq(floor(y_min), ceiling(y_max), by = 0.05)
    ) +
    scale_color_manual(
      values = c("Best Fitness" = "blue", "Worst Fitness" = "red"),
      labels = c("Best Fitness", "Worst Fitness")
    ) +
    labs(
      title = "Fitness Convergence Over Generations",
      x = "Generation",
      y = "Fitness Score",
      caption = paste(log_filename),
      color = "Metric"
    ) +
    theme_minimal(base_size = 13) +
    theme(
      text = element_text(family = "Arial", size = 14),
      legend.position = "bottom",  # Move legend to the right
      legend.text = element_text(size = 8),  # Bigger legend text
      legend.direction = "horizontal",  # Display legend items vertically
      legend.justification = "center",
      legend.title = element_blank(),
      
      
      axis.text.x = element_text(size = 12, hjust = 1),  # Smaller x-axis text
      axis.title.x = element_text(size = 11),  # Change x-axis label size
      
      axis.text.y = element_text(size = 12),  # Bigger Y-axis text
      axis.title.y = element_text(size = 11),  # Change x-axis label size
      
      
      panel.grid.major = element_line(linewidth = 0.5),
      panel.grid.minor = element_line(linewidth = 0.25, linetype = "dotted"),
      plot.caption = element_text(size = 8, hjust = 0.5, face = "italic"),
      plot.title = element_text(size = 15, hjust = 0.5),  # centered title
      
    )
}


plot_and_save_convergence <- function(summary_path, log_dir, plot_dir) {
  # Read the summary file
  summaries <- read_csv(summary_path)
  
  # Create base output directory for convergence plots if it doesn't exist
  convergence_plotting_dir <- file.path(plot_dir, "convergence_plotting")
  if (!dir.exists(convergence_plotting_dir)) {
    dir.create(convergence_plotting_dir, recursive = TRUE)
  }
  
  # Iterate over all log files in the summaries
  for (log_file in summaries$ga_log_filename) {
    # Print the current log file being processed
    print(paste("Processing:", log_file))
    
    # Normalize the log file and extract the task type
    log_path <- file.path(log_dir, log_file)
    log_data <- normalize_ga_log(log_path)
    task <- ifelse(log_data$metadata$config$multi_label, "multi", "binary")
    
    # Define subdirectory for the task type
    task_dir <- file.path(convergence_plotting_dir, task)
    if (!dir.exists(task_dir)) {
      dir.create(task_dir, recursive = TRUE)
    }
    
    # Generate the convergence plot
    plot <- plot_convergence(log_path) # Use your existing plot_convergence function
    
    # Save the plot to the appropriate directory
    plot_file <- file.path(task_dir, paste0(tools::file_path_sans_ext(basename(log_file)), "_convergence.png"))
    ggsave(plot_file, plot = plot, width = 8, height = 6)
    
    print(paste("Saved plot to:", plot_file))
  }
}








plot_and_save_metrics <- function(summary_path, log_dir, plot_dir, metrics = c(
  "accuracy", 
  "fitness_score",
  "average_fitness",
  "macro_avg_f1_score", 
  "weighted_avg_f1_score"
)) {
  # Read the summary file
  summaries <- read_csv(summary_path)
  
  # Create base output directory if it doesn't exist
  metric_plotting_dir <- file.path(plot_dir, "metric_plotting")
  if (!dir.exists(metric_plotting_dir)) {
    dir.create(metric_plotting_dir, recursive = TRUE)
  }
  
  # Iterate over all log files in the summaries
  for (log_file in summaries$ga_log_filename) {
    # Print the current log file being processed
    print(paste("Processing:", log_file))
    
    # Normalize the log file and extract the task type
    log_path <- file.path(log_dir, log_file)
    log_data <- normalize_ga_log(log_path)
    task <- ifelse(log_data$metadata$config$multi_label, "multi", "binary")
    
    # Define subdirectory for the task type
    task_dir <- file.path(metric_plotting_dir, task)
    if (!dir.exists(task_dir)) {
      dir.create(task_dir, recursive = TRUE)
    }
    
    # Generate the plot
    plot <- plot_metrics(log_path, metrics)
    
    # Save the plot to the appropriate directory
    plot_file <- file.path(task_dir, paste0(tools::file_path_sans_ext(basename(log_file)), "_metrics.png"))
    ggsave(plot_file, plot = plot, width = 8, height = 6)
    
    print(paste("Saved plot to:", plot_file))
  }
}


#DIVERSITY PLOTTING AND SUCH 

extract_individuals <- function(log_data, gen) {
  # Get the population size
  population_size <- log_data[["metadata"]][["config"]][["population_size"]]
  
  # Extract individuals for the initial or subsequent generations
  if (gen == 0) {
    prompt_1 <- log_data[["initial_generation"]][["initial_population"]][[1]][["prompt_1"]]
    prompt_2 <- log_data[["initial_generation"]][["initial_population"]][[1]][["prompt_2"]]
  } else {
    prompt_1 <- log_data[["run_generations"]][["population"]][[gen]][["prompt_1"]]
    prompt_2 <- log_data[["run_generations"]][["population"]][[gen]][["prompt_2"]]
  }
  
  # Combine into unordered pairs (tuples)
  combined_prompts <- mapply(function(p1, p2) {
    paste(sort(c(p1, p2)), collapse = " || ")  # Unordered pair
  }, prompt_1, prompt_2)
  
  return(list(
    prompt_1 = prompt_1,
    prompt_2 = prompt_2,
    combined_prompts = combined_prompts
  ))
}

extract_best_individual <- function(log_data, gen) {
  # Extract the best individual for the initial or subsequent generations
  if (gen == 0) {
    prompt_1 <- log_data[["initial_generation"]][["fitness_data"]][["best_individual"]][["individual"]][["prompt_1"]]
    prompt_2 <- log_data[["initial_generation"]][["fitness_data"]][["best_individual"]][["individual"]][["prompt_2"]]
  } else {
    prompt_1 <- log_data[["run_generations"]][["fitness_data"]][["best_individual"]][["individual"]][["prompt_1"]][gen]
    prompt_2 <- log_data[["run_generations"]][["fitness_data"]][["best_individual"]][["individual"]][["prompt_2"]][gen]
  }
  
  return(list(prompt_1 = prompt_1, prompt_2 = prompt_2))
}


# Function to calculate unique individuals for all generations
analyze_diversity <- function(log_data) {
  completed_generations <- log_data[["completed_generations"]]
  
  # Initialize a tibble to store results
  diversity_data <- tibble(
    generation = integer(),
    unique_prompt_1 = integer(),
    unique_prompt_2 = integer(),
    unique_individuals = integer(),
    total_individuals = integer()
  )
  
  # Loop through each generation
  for (gen in 0:completed_generations) {
    # Extract individuals
    extracted <- extract_individuals(log_data, gen)
    unique_prompt_1 <- unique(extracted$prompt_1)
    unique_prompt_2 <- unique(extracted$prompt_2)
    unique_individuals <- unique(extracted$combined_prompts)
    
    diversity_data <- add_row(
      diversity_data,
      generation = gen,
      unique_prompt_1 = length(unique_prompt_1),
      unique_prompt_2 = length(unique_prompt_2),
      unique_individuals = length(unique_individuals),
      total_individuals = length(extracted$combined_prompts)
    )
  }
  
  return(diversity_data)
}

plot_diversity <- function(diversity_data) {
  # Pivot data for plotting
  diversity_long <- diversity_data %>%
    pivot_longer(
      cols = c(unique_prompt_1, unique_prompt_2, unique_individuals),
      names_to = "Type",
      values_to = "Unique_Count"
    )
  
  # Plot diversity
  ggplot(diversity_long, aes(x = generation, y = Unique_Count, color = Type)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    scale_x_continuous(breaks = seq(min(diversity_long$generation), max(diversity_long$generation), by = 3)) +  # Ensure whole number x-axis
    scale_y_continuous(breaks = seq(min(diversity_long$Unique_Count), max(diversity_long$Unique_Count), by = 3)) +  # Ensure whole numbers for y-axis
    
    
    labs(
      title = "Diversity of Prompts and Individuals Over Generations",
      x = "Generation",
      y = "Number of Unique Elements"
      #color = "Type"
    ) +
    theme_minimal(base_size = 13) +  
    theme(
      text = element_text(family = "Arial", size = 14),
      legend.text = element_text(size = 8),  # Bigger legend text
      legend.title = element_blank(),
      
      
      legend.position = "bottom",  # Move legend to the right
      legend.direction = "horizontal",  # Display legend items vertically
      legend.justification = "center",
      axis.text.x = element_text(size = 12, hjust = 1),  # Smaller x-axis text
      axis.title.x = element_text(size = 11),  # Change x-axis label size
      
      plot.title = element_text(size = 15, hjust = 0.5),  # centered title
      
      
      plot.caption = element_text(size = 8, hjust = 0.5, face = "italic"),
      axis.text.y = element_text(size = 12),  # Bigger Y-axis text
      axis.title.y = element_text(size = 11),  # Change x-axis label size
      
      
      panel.grid.major = element_line(linewidth = 0.5),
      panel.grid.minor = element_line(linewidth = 0.25, linetype = "dotted")
    )
}

###SIMILARITY MATRICES

analyze_similarity_of_best <- function(log_data, debug_mode = FALSE) {
  cat("\n=== Running analyze_similarity_of_best ===\n")
  
  # Get the number of completed generations
  completed_generations <- log_data[["completed_generations"]]
  cat("Completed Generations:", completed_generations, "\n")
  
  # Extract the best prompts across generations
  best_prompts <- tibble(
    generation = 0:completed_generations,
    prompt_1 = sapply(0:completed_generations, function(gen) {
      best <- extract_best_individual(log_data, gen)
      if (!is.null(best$prompt_1)) best$prompt_1 else NA
    }),
    prompt_2 = sapply(0:completed_generations, function(gen) {
      best <- extract_best_individual(log_data, gen)
      if (!is.null(best$prompt_2)) best$prompt_2 else NA
    })
  )
  
  # Debugging: Print extracted prompts
  if (debug_mode) {
    cat("\n=== Extracted Best Prompts ===\n")
    print(best_prompts)
  }
  
  # Remove missing values
  best_prompts <- best_prompts %>%
    filter(!is.na(prompt_1) & !is.na(prompt_2))
  
  cat("\nAfter removing NAs, best_prompts shape:", nrow(best_prompts), "rows\n")
  
  # Generate generation labels
  generation_labels <- as.character(best_prompts$generation)
  
  ### ✅ Jaccard Similarity (Character-Based) ###
  cat("\nComputing Jaccard Similarity...\n")
  jaccard_prompt_1 <- 1 - stringdistmatrix(best_prompts$prompt_1, best_prompts$prompt_1, method = "jaccard")
  jaccard_prompt_2 <- 1 - stringdistmatrix(best_prompts$prompt_2, best_prompts$prompt_2, method = "jaccard")
  
  # Debugging: Character-based Jaccard comparisons with clear visualization
  if (debug_mode) {
    cat("\n=== Jaccard Similarity (Character-Level) ===\n")
    for (i in seq_along(best_prompts$prompt_1)) {
      for (j in seq_along(best_prompts$prompt_1)) {
        text1 <- best_prompts$prompt_1[i]
        text2 <- best_prompts$prompt_1[j]
        set1 <- unique(strsplit(text1, "")[[1]])
        set2 <- unique(strsplit(text2, "")[[1]])
        intersection <- length(intersect(set1, set2))
        union <- length(union(set1, set2))
        jaccard_sim <- ifelse(union == 0, 0, intersection / union)
        
        cat("\n🔹 Comparing:\n")
        cat("   ", paste(set1, collapse = " | "), "\n   vs\n")
        cat("   ", paste(set2, collapse = " | "), "\n")
        cat(sprintf("   -> Jaccard Similarity: %.3f\n", jaccard_sim))
      }
    }
  }
  
  ### ✅ Levenshtein Distance (Character-Based) ###
  cat("\nComputing Levenshtein Distance...\n")
  levenshtein_prompt_1 <- stringdistmatrix(best_prompts$prompt_1, best_prompts$prompt_1, method = "lv")
  levenshtein_prompt_2 <- stringdistmatrix(best_prompts$prompt_2, best_prompts$prompt_2, method = "lv")
  
  # Debugging: Character-based Levenshtein comparisons with clear visualization
  if (debug_mode) {
    cat("\n=== Levenshtein Distance (Character-Level) ===\n")
    for (i in seq_along(best_prompts$prompt_1)) {
      for (j in seq_along(best_prompts$prompt_1)) {
        text1 <- best_prompts$prompt_1[i]
        text2 <- best_prompts$prompt_1[j]
        lev_dist <- stringdist(text1, text2, method = "lv")
        
        cat("\n🔹 Comparing:\n")
        cat("   ", paste(strsplit(text1, "")[[1]], collapse = " | "), "\n   vs\n")
        cat("   ", paste(strsplit(text2, "")[[1]], collapse = " | "), "\n")
        cat(sprintf("   -> Levenshtein Distance: %d\n", lev_dist))
      }
    }
  }
  
  ### ✅ Tokenized Jaccard and Levenshtein ###
  cat("\nComputing Tokenized Jaccard and Levenshtein...\n")
  
  # Tokenization function (splits text into unique words)
  tokenize <- function(text) {
    unique(unlist(strsplit(text, "\\s+")))
  }
  
  # Tokenized Jaccard Similarity (1 - Jaccard Distance)
  tokenized_jaccard <- function(text1, text2) {
    set1 <- tokenize(text1)
    set2 <- tokenize(text2)
    if (length(set1) == 0 || length(set2) == 0) return(0)
    1 - stringdist(paste(set1, collapse = " "), paste(set2, collapse = " "), method = "jaccard")
  }
  
  # Tokenized Levenshtein Distance
  tokenized_levenshtein <- function(text1, text2) {
    stringdist(paste(tokenize(text1), collapse = " "), paste(tokenize(text2), collapse = " "), method = "lv")
  }
  
  # Compute similarity matrices
  tokenized_jaccard_prompt_1 <- outer(best_prompts$prompt_1, best_prompts$prompt_1, Vectorize(tokenized_jaccard))
  tokenized_jaccard_prompt_2 <- outer(best_prompts$prompt_2, best_prompts$prompt_2, Vectorize(tokenized_jaccard))
  tokenized_levenshtein_prompt_1 <- outer(best_prompts$prompt_1, best_prompts$prompt_1, Vectorize(tokenized_levenshtein))
  tokenized_levenshtein_prompt_2 <- outer(best_prompts$prompt_2, best_prompts$prompt_2, Vectorize(tokenized_levenshtein))
  
  # Debugging: Tokenized comparisons with visual formatting
  if (debug_mode) {
    cat("\n=== Tokenized Jaccard Similarity ===\n")
    for (i in seq_along(best_prompts$prompt_1)) {
      for (j in seq_along(best_prompts$prompt_1)) {
        words1 <- tokenize(best_prompts$prompt_1[i])
        words2 <- tokenize(best_prompts$prompt_1[j])
        tokenized_jaccard_sim <- tokenized_jaccard(best_prompts$prompt_1[i], best_prompts$prompt_1[j])
        
        cat("\n🔹 Comparing (Tokenized):\n")
        cat("   ", paste(words1, collapse = " || "), "\n   vs\n")
        cat("   ", paste(words2, collapse = " || "), "\n")
        cat(sprintf("   -> Jaccard Similarity: %.3f\n", tokenized_jaccard_sim))
      }
    }
  }
  
  ### ✅ RETURN DATA ###
  return(list(
    prompt_data = best_prompts,  
    jaccard = list(prompt_1 = jaccard_prompt_1, prompt_2 = jaccard_prompt_2),
    levenshtein = list(prompt_1 = levenshtein_prompt_1, prompt_2 = levenshtein_prompt_2),
    tokenized_jaccard = list(prompt_1 = tokenized_jaccard_prompt_1, prompt_2 = tokenized_jaccard_prompt_2),
    tokenized_levenshtein = list(prompt_1 = tokenized_levenshtein_prompt_1, prompt_2 = tokenized_levenshtein_prompt_2)
  ))
}



visualize_similarity_matrices <- function(similarity_results, color_schemes = list()) {
  # ============================
  # Color Scheme Fix:
  # ============================
  # - Jaccard Similarity:
  #   - High similarity (1) = DARKER color (e.g., deeper purple/orange).
  #   - Low similarity (0) = LIGHTER color (closer to white).
  # - Levenshtein Distance:
  #   - Low distance (0) = DARKER color (more similar).
  #   - High distance (large) = LIGHTER color (more different).
  
  # Default color schemes for each metric
  default_colors <- list(
    p1_jaccard = "purple",              # Jaccard for Prompt 1
    p2_jaccard = "orange",              # Jaccard for Prompt 2
    p1_levenshtein = "red",             # Levenshtein for Prompt 1
    p2_levenshtein = "blue",            # Levenshtein for Prompt 2
    p1_tokenized_jaccard = "green",     # Tokenized Jaccard for Prompt 1
    p2_tokenized_jaccard = "cyan",      # Tokenized Jaccard for Prompt 2
    p1_tokenized_levenshtein = "pink",  # Tokenized Levenshtein for Prompt 1
    p2_tokenized_levenshtein = "yellow" # Tokenized Levenshtein for Prompt 2
  )
  
  # Merge provided color schemes with defaults
  colors <- modifyList(default_colors, color_schemes)
  
  # ✅ Extract generation labels from prompt_data (Ensuring proper order)
  generation_labels <- as.character(similarity_results$prompt_data$generation)
  
  # Debugging: Ensure correct labeling
  cat("\n=== Debug: Generation Labels ===\n")
  print(generation_labels)
  
  # ✅ Apply generation labels to all matrices, including tokenized versions
  all_matrices <- list(
    similarity_results$jaccard$prompt_1, similarity_results$jaccard$prompt_2,
    similarity_results$levenshtein$prompt_1, similarity_results$levenshtein$prompt_2,
    similarity_results$tokenized_jaccard$prompt_1, similarity_results$tokenized_jaccard$prompt_2,
    similarity_results$tokenized_levenshtein$prompt_1, similarity_results$tokenized_levenshtein$prompt_2
  )
  
  for (mat in all_matrices) {
    rownames(mat) <- generation_labels
    colnames(mat) <- generation_labels
  }
  
  # Helper function to plot a similarity matrix
  plot_similarity_matrix <- function(matrix, title, fill_color, metric_type) {
    # Check for any NA or Inf values in the matrix
    if (any(is.na(matrix)) || any(is.infinite(matrix))) {
      warning("The similarity matrix contains NA or Inf values. These may appear as blank tiles in the plot.")
    }
    
    # Convert matrix into a data frame for ggplot
    similarity_df <- as.data.frame(as.table(as.matrix(matrix)))
    
    ggplot(similarity_df, aes(Var1, Var2, fill = Freq)) +
      geom_tile(color = "white") +  # White border around tiles
      scale_fill_gradient(
        low = if (metric_type == "jaccard") "white" else fill_color,  # LIGHT for dissimilarity
        high = if (metric_type == "jaccard") fill_color else "white", # DARK for similarity
        name = if (metric_type == "jaccard") "Similarity" else "Distance"
      ) +
      
      #scale_x_discrete(labels = generation_labels) +  # ✅ Apply correct generation labels
      #scale_y_discrete(labels = generation_labels) +  # ✅ Apply correct generation labels
      
      scale_x_discrete(labels = ifelse(seq_along(generation_labels) %% 2 == 0, generation_labels, " ")) +
      scale_y_discrete(labels = ifelse(seq_along(generation_labels) %% 2 == 0, generation_labels, " ")) +
      
      
      labs(
        title = title,
        x = "Generation",
        y = "Generation"
      ) +
      theme_minimal() +
      theme(
        text = element_text(family = "Arial", size = 14),  # Bigger text for all
        plot.title = element_text(family = "Arial", size = 15, hjust = 0.5),  # centered title

        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(hjust = 1),
        panel.grid = element_blank(),
        
        legend.position = "left",  # Move legend to the bottom
        legend.direction = "vertical",  # Make legend horizontal
        legend.title = element_text(size = 9),  # Adjust legend title size if needed
        legend.text = element_text(size = 9)  # Adjust legend text size if needed
      )

  }
  
  # Generate and return plots
  plots <- list(
    # Classic Jaccard Similarity
    p1_jaccard = plot_similarity_matrix(similarity_results$jaccard$prompt_1, "Prompt 1 Jaccard Similarity", colors$p1_jaccard, "jaccard"),
    p2_jaccard = plot_similarity_matrix(similarity_results$jaccard$prompt_2, "Prompt 2 Jaccard Similarity", colors$p2_jaccard, "jaccard"),
    
    # Classic Levenshtein Distance
    p1_levenshtein = plot_similarity_matrix(similarity_results$levenshtein$prompt_1, "Prompt 1 Levenshtein Distance", colors$p1_levenshtein, "levenshtein"),
    p2_levenshtein = plot_similarity_matrix(similarity_results$levenshtein$prompt_2, "Prompt 2 Levenshtein Distance", colors$p2_levenshtein, "levenshtein"),
    
    # Tokenized Jaccard Similarity
    p1_tokenized_jaccard = plot_similarity_matrix(similarity_results$tokenized_jaccard$prompt_1, "Prompt 1 Tokenized Jaccard Similarity", colors$p1_tokenized_jaccard, "jaccard"),
    p2_tokenized_jaccard = plot_similarity_matrix(similarity_results$tokenized_jaccard$prompt_2, "Prompt 2 Tokenized Jaccard Similarity", colors$p2_tokenized_jaccard, "jaccard"),
    
    # Tokenized Levenshtein Distance
    p1_tokenized_levenshtein = plot_similarity_matrix(similarity_results$tokenized_levenshtein$prompt_1, "Prompt 1 Tokenized Levenshtein Distance", colors$p1_tokenized_levenshtein, "levenshtein"),
    p2_tokenized_levenshtein = plot_similarity_matrix(similarity_results$tokenized_levenshtein$prompt_2, "Prompt 2 Tokenized Levenshtein Distance", colors$p2_tokenized_levenshtein, "levenshtein")
  )
  
  return(plots)
}



