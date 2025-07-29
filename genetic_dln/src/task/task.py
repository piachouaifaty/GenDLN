class Task:
    def __init__(
        self,
        layer_1_system_prompt_path,
        layer_2_system_prompt_path,
        layer_2_few_shots_path,
        layer_1_initial_prompts_path,
        layer_2_initial_prompts_path,
        train_dataset_path,
        val_dataset_path
    ):
        self.layer_1_system_prompt_path = layer_1_system_prompt_path
        self.layer_2_system_prompt_path = layer_2_system_prompt_path
        self.layer_2_few_shots_path = layer_2_few_shots_path
        self.layer_1_initial_prompts_path = layer_1_initial_prompts_path
        self.layer_2_initial_prompts_path = layer_2_initial_prompts_path
        self.train_dataset_path = train_dataset_path,
        self.val_dataset_path = val_dataset_path
