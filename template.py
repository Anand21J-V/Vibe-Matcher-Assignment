import os
from pathlib import Path
import logging

# VIBE MATCHER PROJECT TEMPLATE GENERATOR

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# List of directories & files to auto-create
list_of_files = [

    # Main project notebook
    "vibe_matcher.ipynb",

    # Data files
    "data/__init__.py",
    "data/fashion_data.csv",

    # Source code (modular components)
    "src/__init__.py",
    "src/data_prep.py",
    "src/embedder.py",
    "src/matcher.py",
    "src/evaluator.py",

    # Assets for plots, visuals, outputs
    "assets/__init__.py",
    "assets/sample_outputs.txt",
    "assets/latency_plot.png",

    # Supporting files
    ".env",
    "requirements.txt",
    "README.md",
    "LICENSE"
]

# Create directories and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    # Create empty file if not exists
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")

# Summary
logging.info("Project template created successfully. You can now start building your Vibe Matcher.")
