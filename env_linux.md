## Setting up a Virtual Environment on Linux

### 1. Install `python3-venv`

First, ensure that the `venv` module is installed on your system. You can install it using:

```bash
sudo apt-get install python3-venv
```

### 2. Create a Virtual Environment

Navigate to your project directory or where you want to create the virtual environment:

```bash
cd /path/to/your/project/
```

Now, create a virtual environment named 'venv' (or any other name you prefer):

```bash
python3 -m venv venv
```

This will create a directory named 'venv' in your project folder.

### 3. Activate the Virtual Environment

```bash
source venv/bin/activate
```

After activation, your terminal should prefix with `(venv)`, indicating that the virtual environment is active.

### 4. Install Packages

Now you can install packages required for your project using pip. For this, you'll often rely on a `requirements.txt`
file that lists all of your project's dependencies.

For example:

```bash
pip install -r requirements.txt
```

---

## Sample `requirements.txt` for Medical LLM:

```
# Core packages
numpy==1.21.0
pandas==1.3.0
torch==1.9.0

# For data processing
opencv-python==4.5.3
scikit-learn==0.24.2

# LLM related packages (these are hypothetical, ensure you use correct package names and versions)
llama==2.0.0
llama-cpp==1.0.0

# Any other dependencies
...
```

To generate a `requirements.txt` from your current environment in the future, you can use:

```bash
pip freeze > requirements.txt
```

### 5. Deactivating the Virtual Environment

Once you're done with your work in the virtual environment, you can deactivate it:

```bash
deactivate
```
