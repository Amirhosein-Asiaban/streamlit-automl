# AutoML App

A lightweight web application for running common AutoML workflows without writing code. The app is built with Streamlit and uses PyCaret to compare models, train a final model, and generate predictions from a separate CSV dataset.

## Overview

AutoML App is intended for quick experimentation with tabular machine learning problems. You provide a training dataset, choose whether the task is regression or classification, select the target column, and let the application handle model comparison and training.

Once a model has been selected, a second CSV file can be supplied for inference. The resulting predictions can be reviewed in the browser and exported as a CSV file.

The application is deliberately small and focused: it provides a simple interface around PyCaret rather than attempting to replace a full ML development workflow.

## Features

- Supports **regression** and **classification** tasks
- Automatically compares multiple models through PyCaret
- Displays the model comparison leaderboard
- Selects the best-performing model from the comparison
- Accepts CSV files up to **200 MB**
- Lets you choose the target column from the training dataset
- Supports an optional identifier column for prediction data
- Generates a `row_id` automatically when no identifier is selected
- Displays predictions directly in the application
- Allows predictions to be downloaded as a CSV file
- Uses Streamlit session state and caching to keep the interactive workflow manageable
- Includes a Docker configuration for reproducible deployment

## Requirements

### Local environment

- Python **3.10.x**
- pip
- A virtual environment is recommended

The project currently pins the main dependencies to the following versions:

| Package | Version | Role |
| --- | --- | --- |
| Streamlit | 1.38.0 | Web interface |
| PyCaret | 3.3.0 | AutoML and model comparison |
| Pandas | 2.1.4 | Data loading and manipulation |
| NumPy | 1.26.4 | Numerical operations |
| SciPy | 1.11.4 | Scientific computing |
| scikit-learn | 1.4.2 | Machine learning backend |
| joblib | 1.3.2 | Model persistence and utilities |

The complete dependency list is available in `requirements.txt`.

## Installation

### Run locally

Clone the repository and move into the project directory:

```bash
git clone <repository-url>
cd <project-directory>
```

Create and activate a virtual environment:

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

By default, Streamlit serves the application at:

```text
http://localhost:8501
```

## Running with Docker

A Dockerfile is included for environments where you want to run the application without installing the Python dependencies directly on the host machine.

### Build the image

```bash
docker build -t automl-app .
```

### Start the container

```bash
docker run --rm -p 8501:8501 automl-app
```

Open the application in a browser:

```text
http://localhost:8501
```

To run it in the background:

```bash
docker run -d --name automl-app -p 8501:8501 automl-app
```

Stop the container with:

```bash
docker stop automl-app
```

## Dataset Expectations

The application is designed for standard tabular CSV data. Before uploading a dataset, make sure that:

- The training file contains a clearly defined target column.
- Feature columns required for prediction are present in the prediction dataset.
- Training and prediction feature names are consistent.
- The target column does not need to be present in the prediction dataset.
- Categorical and numeric columns are represented in a form that PyCaret can process.
- The target values are present for the rows intended for training.

Column names are converted to strings before being passed to PyCaret to improve compatibility with the underlying pipeline.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

| File | Description |
| --- | --- |
| `app.py` | Main Streamlit application |
| `requirements.txt` | Pinned Python dependencies |
| `Dockerfile` | Docker image configuration |
| `README.md` | Project documentation |

## Docker Image

The Docker image is based on `python:3.10-slim`.

During the image build, the project:

1. Installs the required system build tools.
2. Installs Python dependencies from `requirements.txt`.
3. Copies the application files into the image.
4. Configures the environment for PyCaret and Matplotlib.
5. Exposes Streamlit's default port, `8501`.
6. Starts Streamlit on `0.0.0.0` so the application is accessible from outside the container.

## Technology Stack

| Technology | Purpose |
| --- | --- |
| **Python** | Application and machine learning runtime |
| **Streamlit** | Interactive web interface |
| **PyCaret** | Automated model comparison and training |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical computing |
| **SciPy** | Scientific computing |
| **scikit-learn** | Machine learning algorithms and utilities |
| **joblib** | Model-related utilities |
| **Docker** | Containerization and reproducible execution |

## License

This project is released under the **MIT License**.

---

<div align="center">

Built with Streamlit and PyCaret.

</div>
