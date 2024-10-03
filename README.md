# Sentiment Analysis API with BERT and DistilBERT

A simple sentiment analysis API using BERT and DistilBERT models, packaged in a Docker container for easy deployment.

## Setup Instructions

### Requirements

- **Docker**: Ensure Docker is installed. You can download it from [https://www.docker.com/](https://www.docker.com/).

### Steps to Run

1. **Clone the Repository**:

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Requirements**
    - pip install virtualenv 
    - virtualenv venv
    - source venv/bin/activate
    - pip install -r requirements.txt

3. **Docker Build and Run**
    - docker login
    - docker build -t <name> .
    - docker run -p 5000:5000 <name>
    

dockerized-bert-distilbert/
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
