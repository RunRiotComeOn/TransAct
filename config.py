"""
Configuration file for steering vector experiments
All experimental parameters can be modified here
"""

class Config:
    """Main configuration class for steering experiments"""

    # ============================================
    # Model Configuration
    # ============================================
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    reasoning_model = "Qwen/Qwen2.5-Math-7B-Instruct"

    # Device configuration
    device = "cuda"  # or "cpu"
    torch_dtype = "bfloat16"  # or "float16", "float32"

    # ============================================
    # Dataset Configuration
    # ============================================
    # Choose the ability to test (one of the 6 abilities)
    ability = "mathematical_reasoning"
    # Options:
    # - "mathematical_reasoning"
    # - "advanced_mathematical_reasoning"
    # - "logical_reasoning"
    # - "commonsense_reasoning"
    # - "science_reasoning"
    # - "reading_comprehension"
    # - "multihop_reasoning"

    # Dataset mapping for each ability
    ABILITY_DATASETS = {
        "mathematical_reasoning": "gsm8k",
        "advanced_mathematical_reasoning": "hendrycks_math",
        "logical_reasoning": "winogrande",
        "commonsense_reasoning": "commonsense_qa",
        "science_reasoning": "arc_challenge",
        "reading_comprehension": "boolq",
        "multihop_reasoning": "hotpotqa"
    }

    # Number of samples
    num_extraction_samples = 50  # For extracting steering vectors

    # ============================================
    # Generation Configuration
    # ============================================
    max_length = 2048  # Max input length
    max_new_tokens = 2048  # Max tokens to generate

    # ============================================
    # Output Configuration
    # ============================================
    output_dir = "./steering_outputs"

    # Random seed for reproducibility
    random_seed = 42

    @classmethod
    def get_dataset(cls):
        """Get the dataset name for current ability"""
        return cls.ABILITY_DATASETS.get(cls.ability, "gsm8k")

    @classmethod
    def get_experiment_name(cls):
        """Generate experiment name if not specified"""
        if cls.experiment_name:
            return cls.experiment_name

        dataset = cls.get_dataset()
        model_size = "7b" if "7B" in cls.base_model else "3b"
        return f"{cls.ability}_{dataset}_alpha{cls.fixed_alpha}_{model_size}"

    @classmethod
    def display(cls):
        """Display current configuration"""
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        print(f"Base Model: {cls.base_model}")
        print(f"Reasoning Model: {cls.reasoning_model}")
        print(f"Ability: {cls.ability}")
        print(f"Dataset: {cls.get_dataset()}")
        print(f"Extraction Samples: {cls.num_extraction_samples}")
        print(f"Output Dir: {cls.output_dir}/{cls.get_experiment_name()}")
        print("="*60 + "\n")
