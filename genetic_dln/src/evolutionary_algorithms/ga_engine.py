import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
from typing import Any, Dict, Tuple

from genetic_dln.src.constants import constants
from genetic_dln.src.evolutionary_algorithms.genetic_operations.population_initialization import PopulationInitializer
from genetic_dln.src.evolutionary_algorithms.genetic_operations.mutator import Mutator
from genetic_dln.src.evolutionary_algorithms.genetic_operations.crossover import Crossover
from genetic_dln.src.evolutionary_algorithms.genetic_operations.replacement import replacement
from genetic_dln.src.evolutionary_algorithms.genetic_operations.selection import Selection
from genetic_dln.src.evolutionary_algorithms.genetic_operations.fitness import FitnessEvaluator
from genetic_dln.src.evolutionary_algorithms.loggers.crossover_logger import CrossoverLogger
from genetic_dln.src.evolutionary_algorithms.loggers.fitness_cache_logger import FitnessCacheLogger
from genetic_dln.src.evolutionary_algorithms.loggers.fitness_logger import FitnessLogger
from genetic_dln.src.evolutionary_algorithms.loggers.ga_logger import GALogger
from genetic_dln.src.evolutionary_algorithms.loggers.mutation_logger import MutationLogger
from genetic_dln.src.evolutionary_algorithms.loggers.population_initialization_logger import \
    PopulationInitializationLogger
from genetic_dln.src.evolutionary_algorithms.loggers.replacement_logger import ReplacementLogger
from genetic_dln.src.evolutionary_algorithms.loggers.selection_logger import SelectionLogger
from genetic_dln.src.input_loader.input_loader import InputLoader
from genetic_dln.src.models.llm import LLM
import random
from datetime import datetime
import platform
import psutil


class GAEngine:
    def __init__(self, config, task):
        """
        Initialize the GA Engine with configuration and loggers.

        Args:
            config (dict): Configuration for the GA process.
        """
        self.config = config
        self.task = task
        self.multi_label = config["multi_label"]
        self.generation_count = 0
        self.population = []
        self.llm_client = LLM(number_of_workspaces=10)

        self.input_loader = InputLoader()
        self.hyperparameters = self.input_loader.read_hyperparameters(constants.HYPERPARAMETERS_PATH)
        self.llm_configs = self.hyperparameters["run_config"]["llm_configs"]

        # Initialize loggers
        self.ga_logger = GALogger(log_dir="logs")
        self.population_logger = PopulationInitializationLogger(log_dir="logs")
        self.mutation_logger = MutationLogger(log_dir="logs")
        self.crossover_logger = CrossoverLogger(log_dir="logs")
        self.replacement_logger = ReplacementLogger(log_dir="logs")
        self.fitness_logger = FitnessLogger(log_dir="logs")
        self.selection_logger = SelectionLogger(log_dir="logs")
        self.cache_logger = FitnessCacheLogger(log_dir="logs")  # Add cache logger
        self.evaluator = FitnessEvaluator(multi_label=self.multi_label, task=self.task)
        # Fitness cache (shared across GA run)
        self.fitness_cache: Dict[Tuple[str, str], Any] = {}
        self.number_of_workspaces = self.config["number_of_workspaces"]

    def initialize_population(self):
        """
        Initialize the population for the GA process.
        """
        llm_config = self.llm_configs["population_initialization"]
        temperature = llm_config.get("temperature", 0.7)

        initializer = PopulationInitializer(
            layer1_file=constants.layer1_file_binary,
            layer2_file=constants.layer2_file_binary,
            population_size=self.config["population_size"],
            augment_with_llm=self.config["augment_with_llm"],
            logger=self.population_logger,
            temperature=temperature
        )
        self.population, augmentation_temperature = initializer.initialize_population()

        # Retrieve logs from the individual logger
        population_logs = self.population_logger.get_logs()

        # Log the initialization to the GA logger
        self.ga_logger.log_population_initialization(
            generation_id=0,  # Generation 0 for initialization
            population_data=population_logs,
            augmentation_temperature=augmentation_temperature
        )

        # REMOVED IN PRODUCTION
        self.ga_logger.save_logs()

        # Evaluate fitness for Generation 0
        self.evaluate_fitness()

        # Removed in production
        self.ga_logger.save_logs()

    def calculate_fitness_for_individual(self, individual, index):
        fitness_score = self.evaluator.calculate_fitness(
            individual=(individual["prompt_1"], individual["prompt_2"]),
            generation_id=self.generation_count,
            individual_id=individual["id"],
            logger=self.fitness_logger,  # Log fitness details
            cache=self.fitness_cache,
            cache_logger=self.cache_logger,
            index=index
        )
        return individual, fitness_score

    def evaluate_fitness(self):
        """
        Evaluate fitness for the current population.
        """
        # TODO CHECK IF FITNESS ALREADY CALCULATED FOR TUPLE
        # Use ThreadPoolExecutor for concurrent execution
        with ThreadPoolExecutor(max_workers=self.number_of_workspaces) as executor:
            # Submit tasks for all individuals
            futures = {executor.submit(self.calculate_fitness_for_individual, individual, index): individual for index, individual in
                       enumerate(self.population)}

            # Collect results as they complete
            for future in futures:
                individual, fitness_score = future.result()
                individual["fitness_score"] = fitness_score

        # for individual in self.population:
        #     _, individual["fitness_score"] = self.calculate_fitness_for_individual(individual)

        # Summarize and log generation-level fitness
        summary = self.fitness_logger.summarize_generation(generation_id=self.generation_count)
        self.fitness_logger.save_logs(summary)

        # Update GA log with fitness data
        self.ga_logger.data["generations"][-1]["fitness_data"] = summary

        # Reset the fitness logger for the next generation
        self.fitness_logger.reset()

    def perform_selection(self):
        """
        Perform selection on the current population.
        """
        # Prepare population and fitness data for selection
        population_for_selection = [
            {"id": ind["id"], "individual": (ind["prompt_1"], ind["prompt_2"])}
            for ind in self.population
        ]
        fitness_scores = [ind["fitness_score"] for ind in self.population]

        # Initialize the selector
        selector = Selection(
            elitism_k=self.config.get("elitism_k", 0),
            logger=self.selection_logger
        )

        # Perform selection
        selected = selector.select(
            population=population_for_selection,
            fitness_scores=fitness_scores,
            strategy=self.config.get("selection_strategy", "roulette"),
            **self.config.get("selection_params", {})
        )

        # Map selected individuals back to the full population structure
        selected_population = []
        for selected_ind in selected:
            # Find the matching individual in the current population
            original_ind = next(ind for ind in self.population if ind["id"] == selected_ind["id"])

            # Preserve original structure, including fitness score
            selected_population.append({
                "id": selected_ind["id"],  # Preserve ID initially for logging/debugging
                "prompt_1": original_ind["prompt_1"],
                "prompt_2": original_ind["prompt_2"],
                "fitness_score": original_ind["fitness_score"],  # Map fitness back
                "source": selected_ind["source"]  # Use the already tagged source
            })

        # Reset IDs for the selected population to maintain consistency
        for idx, individual in enumerate(selected_population):
            individual["id"] = f"I{idx + 1:02d}"  # Reset IDs in the format I01, I02, etc.

        # Update the internal population in GAEngine
        self.population = selected_population

        # Log selection details to GA logger
        selection_logs = self.selection_logger.get_logs()
        self.ga_logger.data["generations"][-1]["selection_data"] = selection_logs  # Add to GA Logger

        # Reset the selection logger for the next generation
        self.selection_logger.reset()

        # Log the final selected population with reset IDs
        self.ga_logger.data["generations"][-1]["population_after_selection"] = [
            {"id": individual["id"], "fitness_score": individual["fitness_score"],
             "prompt_1": individual["prompt_1"], "prompt_2": individual["prompt_2"]}
            for individual in self.population
        ]

        # Save the updated logs
        self.ga_logger.save_logs()

        return self.population

    def perform_crossover(self):
        """
        Perform crossover on the selected population using parallel execution.
        """
        # Filter non-elite individuals for crossover
        selected_individuals = [ind for ind in self.population if ind["source"] == "selected"]

        # Shuffle the selected individuals to avoid positional bias
        random.shuffle(selected_individuals)

        llm_config = self.llm_configs["crossover"]
        crossover_temperature = llm_config.get("temperature", 0.7)

        # Initialize the crossover module
        crossover = Crossover(
            logger=self.crossover_logger
        )

        # Generate all parent pairs (step by 2)
        pairs = []
        for i in range(0, len(selected_individuals) - 1, 2):
            if random.random() < self.config.get("crossover_rate", 0.8):
                pairs.append((selected_individuals[i], selected_individuals[i + 1], len(self.population) + i))

        # Parallel crossover operation
        # Define crossover_pair to accept both pair and index
        def crossover_pair(pair_with_index):
            pair, idx = pair_with_index
            parent1, parent2, id_base = pair

            prompt_1_result = crossover.perform_crossover(
                parent_1=parent1["prompt_1"],
                parent_2=parent2["prompt_1"],
                crossover_type=self.config.get("crossover_type", "single_point"),
                temperature=crossover_temperature,
                idx=idx
            )
            prompt_2_result = crossover.perform_crossover(
                parent_1=parent1["prompt_2"],
                parent_2=parent2["prompt_2"],
                crossover_type=self.config.get("crossover_type", "single_point"),
                temperature=crossover_temperature,
                idx=idx
            )
            return [
                {
                    "id": f"I{id_base:02d}",
                    "prompt_1": prompt_1_result["child_1"],
                    "prompt_2": prompt_2_result["child_1"],
                    "fitness_score": None,
                    "source": "offspring"
                },
                {
                    "id": f"I{id_base + 1:02d}",
                    "prompt_1": prompt_1_result["child_2"],
                    "prompt_2": prompt_2_result["child_2"],
                    "fitness_score": None,
                    "source": "offspring"
                }
            ]

        # Execute crossover in parallel
        with ThreadPoolExecutor(max_workers=self.number_of_workspaces) as executor:
            results = executor.map(crossover_pair, [(pair, idx) for idx, pair in enumerate(pairs)])
        # Flatten results
        offspring_population = [child for result in results for child in result]
        self.population.extend(offspring_population)

        # Logging and tracking
        self.ga_logger.data["generations"][-1]["population_after_crossover"] = [
            {
                "id": individual["id"],
                "fitness_score": individual["fitness_score"],
                "prompt_1": individual["prompt_1"],
                "prompt_2": individual["prompt_2"],
                "source": individual["source"]
            }
            for individual in self.population
        ]

        self.ga_logger.data["generations"][-1]["crossover_data"] = self.crossover_logger.get_logs()
        self.ga_logger.save_logs()
        self.crossover_logger.reset()

    def perform_mutation(self):
        """
        Perform mutation on the current population based on the mutation rate and mutate_elites flag.
        Executes mutation in parallel and tracks index for downstream processing.
        """
        mutate_elites = self.config.get("mutate_elites", False)
        mutation_rate = self.config.get("mutation_rate", 0.2)
        mutation_temperature = self.llm_configs["mutation"].get("temperature", 0.7)

        mutator = Mutator(logger=self.mutation_logger)

        # Decide which individuals to mutate
        individuals_to_mutate = self.population if mutate_elites else [
            ind for ind in self.population if ind["source"] in {"selected", "offspring"}
        ]

        # Mark elites as not mutated if excluded
        if not mutate_elites:
            for ind in self.population:
                if ind["source"] == "elite":
                    ind["mutated"] = False

        # Mutation function with index
        def mutate_individual(args):
            individual, idx = args
            mutation_logs = []
            mutated = False

            if random.random() < mutation_rate:
                result = mutator.mutate_prompt(
                    prompt=individual["prompt_1"],
                    mutation_type=self.config.get("mutation_type", "semantic"),
                    temperature=mutation_temperature,
                    idx=idx
                )
                individual["prompt_1"] = result["mutated_prompt"]
                individual["fitness_score"] = None
                mutation_logs.append(result)
                mutated = True

            if random.random() < mutation_rate:
                result = mutator.mutate_prompt(
                    prompt=individual["prompt_2"],
                    mutation_type=self.config.get("mutation_type", "semantic"),
                    temperature=mutation_temperature,
                    idx=idx
                )
                individual["prompt_2"] = result["mutated_prompt"]
                individual["fitness_score"] = None
                mutation_logs.append(result)
                mutated = True

            individual["mutated"] = mutated
            return mutation_logs


        # Run mutations in parallel
        with ThreadPoolExecutor(max_workers=self.number_of_workspaces) as executor:
            results = executor.map(mutate_individual, [(indvs, idx) for idx, indvs in enumerate(individuals_to_mutate)])

        # Flatten and collect all mutation logs
        mutation_data = [log for result in results for log in result]

        # Logging
        self.ga_logger.data["generations"][-1]["mutation_data"] = mutation_data
        self.ga_logger.data["generations"][-1]["population_after_mutation"] = [
            {
                "id": ind["id"],
                "fitness_score": ind["fitness_score"],
                "prompt_1": ind["prompt_1"],
                "prompt_2": ind["prompt_2"],
                "source": ind["source"],
                "mutated": ind.get("mutated", False)
            }
            for ind in self.population
        ]

        self.ga_logger.save_logs()
        self.mutation_logger.reset()

    def perform_replacement(self):
        """
        Perform replacement to determine the next generation's population.
        """
        # Evaluate fitness for the current population
        self.evaluate_fitness()

        # Prepare population with fitness for replacement
        population_with_fitness = [
            {
                "id": ind["id"],
                "individual": (ind["prompt_1"], ind["prompt_2"]),
                "fitness": ind["fitness_score"]
            }
            for ind in self.population
        ]

        # Perform replacement using the replacement module
        new_population_data = replacement(
            population_with_fitness=population_with_fitness,
            target_size=self.config["population_size"],
            logger=self.replacement_logger
        )

        # Map the new population back to the GAEngine structure
        self.population = [
            {
                "id": ind["id"],
                "prompt_1": ind["individual"][0],
                "prompt_2": ind["individual"][1],
                "fitness_score": ind["fitness"],
                "source": "carried_over",  # Mark as carried over to the next generation
            }
            for ind in new_population_data
        ]

        # Log replacement data into GA logger
        self.ga_logger.data["generations"][-1]["replacement_data"] = self.replacement_logger.get_logs()[-1]


        # Save (REMOVE IN PRODUCTION)
        self.replacement_logger.save_logs()
        # reset logger
        self.replacement_logger.reset()

    def initialize_cache(self):
        """
        Initialize the fitness cache for the GA run.
        """
        cache_file: str = constants.CACHE_FOLDER + "/" + self.config["cache_file"] + ".json"
        cache_json: Dict[str, Any] = {}
        # Load the cache file if it exists
        if os.path.exists(cache_file):
            with open(cache_file, "r") as cache_fp:
                cache_json = json.load(cache_fp)
        for key, value in cache_json.items():
            self.fitness_cache[ast.literal_eval(key)] = value
    
    def save_cache(self):
        """
        Save the fitness cache to a file.
        """
        cache_file: str = constants.CACHE_FOLDER + "/"  + self.config["cache_file"] + ".json"
        cache_json: Dict[str, Any] = {}
        for key, value in self.fitness_cache.items():
            cache_json[str(key)] = value
        ordered_cache = dict(sorted(cache_json.items()))
        with open(cache_file, "w") as cache_fp:
            json.dump(ordered_cache, cache_fp, indent=4)

    def get_system_info(self):
        """
        Retrieve system information for logging purposes.

        Returns:
            dict: Dictionary containing system information.
        """
        return {
            "system": platform.system(),
            "system_version": platform.version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "cores": psutil.cpu_count(logical=True),
            "memory": f"{round(psutil.virtual_memory().total / (1024 ** 3), 2)} GB"
        }



    def get_best_fitness_from_logs(self):
        """
        Retrieve the best fitness score from the latest generation's logs.

        Returns:
            float: Best fitness score of the latest generation, or -inf if unavailable.
        """
        if self.ga_logger.data["generations"]:
            last_gen = self.ga_logger.data["generations"][-1]
            if "fitness_data" in last_gen and "best_individual" in last_gen["fitness_data"]:
                return last_gen["fitness_data"]["best_individual"]["fitness_score"]
        return -float("inf")

    def get_stagnant_generations_from_logs(self, max_stagnant_generations):
        """
        Calculate the number of stagnant generations based on logs.

        Args:
            max_stagnant_generations (int): Number of generations to consider for stagnation.

        Returns:
            int: Count of consecutive stagnant generations with no improvement in best fitness.
        """
        # Ensure we have enough generations to check stagnation
        if len(self.ga_logger.data["generations"]) < max_stagnant_generations + 1:
            return 0  # Not enough data to assess stagnation

        # Extract fitness scores for the last `max_stagnant_generations + 1` generations
        fitness_scores = [
            gen["fitness_data"]["best_individual"]["fitness_score"]
            for gen in self.ga_logger.data["generations"][-(max_stagnant_generations + 1):]
            if "fitness_data" in gen and "best_individual" in gen["fitness_data"]
        ]

        # Count how many consecutive generations show no improvement
        stagnant_count = 0
        for i in range(1, len(fitness_scores)):
            if fitness_scores[i] <= fitness_scores[i - 1]:  # No improvement
                stagnant_count += 1
            else:
                stagnant_count = 0  # Reset on improvement

        return stagnant_count

    def run_generation(self):
        """
        Run a single generation of the GA process.
        """

        print("GENERATION: ", self.generation_count)
        print("Starting population:", self.population)
        print(f"{len(self.population)} individuals")

        # Log the starting population for the generation
        self.ga_logger.log_generation(
            generation_id=self.generation_count,
            population=[
                {
                    "id": individual["id"],
                    "prompt_1": individual["prompt_1"],
                    "prompt_2": individual["prompt_2"],
                    "fitness_score": individual["fitness_score"],
                    "source": individual["source"],
                }
                for individual in self.population
            ]
        )

        # Perform selection
        self.perform_selection()

        print("Population after selection:", self.population)
        print(f"{len(self.population)} individuals")

        # Perform crossover
        self.perform_crossover()

        print("Population after crossover:", self.population)
        print(f"{len(self.population)} individuals")

        # Perform mutation
        self.perform_mutation()

        print("Population after selection:", self.population)
        print(f"{len(self.population)} individuals")

        # Perform replacement
        self.perform_replacement()

        # save cache after each run
        self.save_cache()

        print("Population after replacement:", self.population)
        print(f"{len(self.population)} individuals")

        # Reset IDs for the next generation
        for idx, individual in enumerate(self.population):
            individual["id"] = f"I{idx + 1:02d}"

        print(f"End of generation {self.generation_count} population:", self.population)

        # Save logs at the end of the generation
        # keep them as intermediary logs
        self.ga_logger.save_logs()

    def run(self):
        """
        Run the full GA process for the configured number of generations.
        """
        start_time = datetime.now()
        print("START TIME: ", start_time)

        # Gather system information
        system_info = self.get_system_info()

        print("SYSTEM INFO: ", system_info)

        self.initialize_cache()
        print("---------")
        print("Cache Initialized")
        print("---------")

        self.initialize_population()
        print("Starting population:", self.population)
        print("POPULATION INITIALIZED, GENERATION", str(self.generation_count))

        max_generations = self.config["generations"]
        max_stagnant_generations = self.config["early_stopping"]["max_stagnant_generations"]
        #will only start checking stagnation after at least max_stagnant_generations

        fitness_goal = self.config["early_stopping"]["fitness_goal"]

        stopped_early = False
        stop_reason = ""

        for _ in range(max_generations):
            self.generation_count += 1
            print("-----NEW RUN-----")
            print("---------")
            print("BEGINNING RUN FOR GENERATION {}".format(self.generation_count))
            self.run_generation()
            print("GENERATION " + str(self.generation_count) + "RUN COMPLETE")
            print("Final population :", self.population)

            # Lookup best fitness from the logs
            best_fitness = self.get_best_fitness_from_logs()

            # Skip stagnation checks for the first few generations
            if self.generation_count >= max_stagnant_generations:
                stagnant_generations = self.get_stagnant_generations_from_logs(max_stagnant_generations)
                print(f"Best Fitness: {best_fitness}, Stagnant Generations: {stagnant_generations}")

                # Check stopping conditions
                if best_fitness >= fitness_goal:
                    stopped_early = True
                    stop_reason = f"Stopped early as fitness goal {fitness_goal} is reached."
                    print(stop_reason)
                    break
                if stagnant_generations >= max_stagnant_generations:
                    stopped_early = True
                    stop_reason = f"Stopped early due to {max_stagnant_generations} stagnant generations."
                    print(stop_reason)
                    break

        # Save cache logs at the end of the run
        self.cache_logger.save_logs()

        end_time = datetime.now()
        print("END TIME: ", end_time)
        total_runtime = (end_time - start_time).total_seconds() / 60  # Runtime in minutes

        # Log the runtime and configuration to the GA logger
        self.ga_logger.data["early_stopping"] = {
            "status": stopped_early,
            "reason": stop_reason
        }

        self.ga_logger.data["runtime"] = f"{total_runtime:.4f} minutes"  # Format to 4 decimal places
        self.ga_logger.data["system_info"] = system_info  # Log system information or else runtime is meaningless
        #although adding both is redundant (config is contained in hyperparams), need it for easier post-processing
        self.ga_logger.data["config"] = self.config
        self.ga_logger.data["hyperparameters"] = self.hyperparameters
        self.ga_logger.data["ga_log_filename"] = self.ga_logger.log_file  # Add the log filename


        # save full ga_log
        self.ga_logger.save_logs()

        # update score cache file
        self.save_cache()

        print(f"GA run completed in {total_runtime:.2f} minutes.")
        print(f"GA log saved to {self.ga_logger.log_file}")

