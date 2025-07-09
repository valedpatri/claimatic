# Claims Processing API

FastAPI service for analyzing, categorizing, and managing customer claims with AI-powered processing.

## Key Features

- **Multi-stage Claim Processing**:
  - Automatic translation (RU→EN)
  - Sentiment analysis (API Layer)
  - Category detection (Ollama)
  - Priority ranking

- **Database Operations**:
  - Async SQLite storage
  - Claim status tracking
  - Time-based queries

- **Monitoring**:
  - Health check endpoint
  - Recent claims dashboard
  - Detailed logging

## Tech Stack

| Component          | Technology                 |
|--------------------|----------------------------|
| Framework          | FastAPI                    |
| Async HTTP         | aiohttp                    |
| Database           | aiosqlite                  |
| NLP               | HuggingFace/Ollama/Mistral |
| Config Management  | Pydantic Settings          |
| Logging           | Loguru                     |

## Installation

1. Clone repository:
   ```bash
   gh repo clone valedpatri/claimatic

   poetry install --no-root

   uvicorn main:app --reload
    ```

graph TD
    A[Client] --> B[POST /add-claim]
    B --> C{Translation?}
    C -->|RU| D[LLMTranslator]
    C -->|EN| E[Sentiment Analysis]
    D --> E
    E --> F[Category Detection]
    F --> G[Database Storage]
    G --> H[ClaimRank Response]


[<img src="docs/images/img_01.png" width="1000"/>]()

[<img src="docs/images/img_02.jpeg" width="300"/>]()

[<img src="docs/images/img_03.png" width="1000"/>]()

[<img src="docs/images/img_04.png" width="1000"/>]()
