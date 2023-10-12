from abc import ABC, abstractmethod


class MedicalLLM(ABC):

    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.model = self.load_model(model_path)
        self.data = self.load_data(data_path)

    @abstractmethod
    def load_data(self, data_path):
        """
        Load medical data from the specified path.

        Args:
        - data_path (str): Path to the medical data.

        Returns:
        - data (object): Loaded medical data.
        """
        pass

    @abstractmethod
    def load_model(self, model_path):
        """
        Load the Llama 2 model from the specified path.

        Args:
        - model_path (str): Path to the Llama 2 model.

        Returns:
        - model (object): Loaded Llama 2 model.
        """
        pass

    @abstractmethod
    def preprocess_data(self):
        """
        Preprocess and sanitize the loaded medical data.
        """
        pass

    @abstractmethod
    def generate_prompt(self, data_sample):
        """
        Generate a suitable prompt for the LLM based on a data sample.

        Args:
        - data_sample (object): A sample of the medical data.

        Returns:
        - prompt (str): Generated prompt for LLM.
        """
        pass

    @abstractmethod
    def interpret_data(self, prompt):
        """
        Use Llama 2 model to interpret the data based on the provided prompt.

        Args:
        - prompt (str): LLM prompt.

        Returns:
        - interpretation (object): Interpretation result from Llama 2 model.
        """
        pass

    @abstractmethod
    def post_process(self, interpretation):
        """
        Post-process the interpretation result into a structured format, like JSON.

        Args:
        - interpretation (object): Interpretation result from Llama 2 model.

        Returns:
        - structured_data (object): Structured interpretation result.
        """
        pass

    @abstractmethod
    def validate_output(self, structured_data):
        """
        Validate the structured output against predefined benchmarks or gold standards.

        Args:
        - structured_data (object): Structured interpretation result.

        Returns:
        - is_valid (bool): Whether the output meets the standards.
        """
        pass