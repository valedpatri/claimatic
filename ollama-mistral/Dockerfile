FROM ollama/ollama

COPY ./Modelfile /root/.ollama/models/Modelfile
COPY entrypoint.sh /entrypoint.sh

RUN apt-get update && apt-get install -y curl && chmod +x /entrypoint.sh

EXPOSE 11434
ENTRYPOINT ["/entrypoint.sh"]
