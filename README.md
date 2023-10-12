# Medical LLM🚀🪐

Welcome to the Medical LLM repository, your premier gateway to a structured and meticulously designed pipeline exclusively crafted for large language models (LLMs) based medical projects. In a domain where precision, accuracy, and reliability are paramount, leveraging the advanced capabilities of LLMs in medical data interpretation and analysis becomes a pivotal element. This project roots deeply into ensuring medical data, including but not limited to medical imaging and tabular reports, can be efficiently processed, interpreted, and utilized using LLMs, particularly focusing on the Llama 2 model.

Harnessing the power of LLMs, this repository aspires to create a pipeline that not only adeptly parses through vast and multifaceted medical data but also curates and converts it into a structured format, ready for in-depth analysis and utilization in various medical applications.

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

This repository contains:

1. Data Pre-processing Tools:
- Utilities for sanitizing and cleaning medical data.
- Tools for segregating and classifying data into types (e.g., imaging, tabular, textual).

2. LLM Prompt Engineering:
- Modules for crafting precise prompts tailored for medical data interpretation.
- Sample prompts showcasing effective interaction with Llama 2.

3. CSV Parsing Mechanism:
- Robust parsers for handling extensive medical CSV datasets.
- Features for transforming CSV data into LLM-compatible formats.

4. llama.cpp Adaptation:
- Customized version of llama.cpp tailored for medical data.
- Documentation detailing the integration and modifications made for the project.

5. Fine-tuning and Training Scripts:
- Scripts and notebooks for training and fine-tuning Llama 2 on specific medical datasets.
- Checkpoints and model weights for various iterations and enhancements.

7. Output Structuring Tools:
- Utilities leveraging Llama 2’s grammar features to transform outputs into JSON format.
- Validation scripts to ensure the correctness and accuracy of structured outputs.

7. Security Protocols:
- Guidelines and tools to ensure that the data processing adheres to privacy and security regulations in the medical domain.

9. Documentation and Guides:
- Comprehensive user guide detailing how to use the pipeline from start to finish.
- Background information on LLM in the medical sector and the significance of Llama 2.

9. Sample Data and Use Cases:
- A selection of sample medical datasets for testing and demonstration purposes.
- Illustrative use cases showcasing the real-world application of the pipeline.

11. Contributor Guidelines and Code of Conduct:
- Information for potential contributors on how to get involved.
- Community guidelines to foster a respectful and collaborative environment.

11. Dependencies and Setup Instructions:
- A detailed list of software dependencies and third-party libraries.
- Step-by-step setup instructions for a seamless project setup.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Workflow](#workflow)
- [Maintainers](#maintainers)
- [NotionPage](#notionpage)
- [Contributing](#contributing)
- [License](#license)

## Background
### LLM in Medical Sector: A Glimpse of Relevance
In the sprawling field of medical science, dealing with colossal amounts of data is routine. This data, ranging from medical images to intricate reports, holds within it the potential to unravel novel insights, support diagnostic processes, and fortify medical research. Large Language Models like Llama 2 propose a prolific methodology to tap into this potential by offering capabilities like natural language understanding, generation, and, crucially for our purposes, transformation of unstructured data into structured formats.

### Why Llama 2?
[Llama 2](https://ai.meta.com/llama/) is a potent LLM developed by Meta AI, offering a splendid array of features such as enhanced natural language understanding and expansive knowledge retrieval. Its unique capability to utilize grammar for refining output into predefined formats, such as JSON, aligns seamlessly with our objective of structuring medical data for further analysis and application.

### Leveraging llama.cpp
Our project strategically employs [llama.cpp](https://github.com/ggerganov/llama.cpp), an adept C++ adaptation of the Llama model. This integration not only allows us to exploit the capabilities of Llama 2 but also facilitates a robust, efficient, and performant pipeline, tailored to navigate through the extensive and diverse arrays of medical data.


## Install

### Prerequisites
#### Hardware recommendations
* TBD


### Setting up the Environment
* Please refer to [env_linux.md](env_linux.md)


## Usage

### Data Preparation
1. TBD
    
## Workflow

[Medical LLM Backlog]()

[Milestones]()


## Milestones: Crafting the Pipeline from Scratch
Embarking on this exciting journey, our path is demarcated into strategic milestones, each propelling us a step closer towards a refined and efficient pipeline.

### Milestone 1: Data Acquisition and Pre-processing
Data Collection: Gather a comprehensive and diverse array of medical data.
Data Sanitization: Implement procedures to clean and sanitize data, ensuring it is free from inconsistencies and anomalies.
Data Segregation: Classify and segregate data into respective types (e.g., imaging, tabular, textual) for specialized processing.
### Milestone 2: Crafting LLM Prompts and Engineering
Prompt Design: Develop precise and targeted prompts to instruct the LLM adeptly in data interpretation.
Data Conversion: Create mechanisms to convert medical data into formats compatible for LLM interpretation.
Model Training: Train and fine-tune the LLM on medical data to enhance its predictive and analytical capabilities.
### Milestone 3: Implementing llama.cpp
Integration: Integrate llama.cpp into the pipeline ensuring smooth interaction with the Llama 2 model.
Adaptation: Customize llama.cpp to cater to the specialized requirements of medical data processing.
Optimization: Ensure that the implementation is optimized for enhanced performance and accuracy in medical data interpretation.
### Milestone 4: Output Structuring and Post-processing
Data Structuring: Employ Llama 2’s grammar capabilities to structure output into predefined formats like JSON.
Validation: Implement procedures to validate the structured output against predefined benchmarks for accuracy and reliability.
Security Compliance: Ensure that the pipeline adheres to security and privacy regulations pertinent to medical data.
### Milestone 5: Validation and Deployment
Testing: Engage in thorough testing of the pipeline across various datasets to validate its efficacy and accuracy.
Feedback Iteration: Implement feedback mechanisms for continuous improvement of the pipeline.
Deployment: Deploy the pipeline for real-world application, ensuring it is scalable, robust, and reliable in diverse operational environments.


## Maintainers

[@Jeff](https://github.com/Ultimate-Storm)


## NotionPage

See this [Meeting link]([https://www.notion.so/Radiology-404e0bcd8c364c45846d5320ea8c10bf](https://www.notion.so/LLM-Weekly-Meeting-1289b5cb4539443a95186d0854f18107))

## Contributing

Feel free to dive in! [Open an issue]() or submit PRs.

Before creating a pull request, please take some time to take a look at
our [wiki page](), to ensure good code quality and sufficient
documentation. Don't need to follow all of the guidelines at this moment, but it would be really helpful!

### Contributors

This project exists thanks to all the people who contribute.
TUD LLM core team:
[@Dyke]()
[@Isabella]()
[@Nafisa]()
[@Radhika]()

## License

[MIT](LICENSE)
