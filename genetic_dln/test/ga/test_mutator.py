from genetic_dln.src.evolutionary_algorithms.genetic_operations.mutator import Mutator
from genetic_dln.src.evolutionary_algorithms.loggers.mutation_logger import MutationLogger
from genetic_dln.src.models.llm import LLM


def test_mutator():
    """
    Test the Mutator class with various mutation types and prompts.
    """
    # Initialize the logger
    mutation_logger = MutationLogger(log_dir="../../src/evolutionary_algorithms/logs")
    #log filename will have timestamp of logger init

    # Initialize the LLM client and Mutator with the logger
    mutator = Mutator(logger=mutation_logger)

    prompts = ["Summarize the following text.", "Write down step-by-step why this can be fair or unfair.",
               "Based on the description above, is the following fair or unfair?"]
    mutation_types = ["random", "swap", "scramble", "inversion", "deletion", "insertion", "semantic", "syntactic"]

    for prompt in prompts:
        print(f"\nOriginal Prompt: {prompt}")
        for mutation_type in mutation_types:
            print(f"\n--- Mutation Type: {mutation_type} ---")
            result = mutator.mutate_prompt(prompt, mutation_type)
            print(f"Mutation Instruction: {result['mutation_instruction']}")
            print(f"Initial Prompt: {result['initial_prompt']}")
            print(f"Mutated Prompt: {result['mutated_prompt']}")
            print("-----------------------")

    mutation_logger.save_logs()


if __name__ == "__main__":
    test_mutator()
