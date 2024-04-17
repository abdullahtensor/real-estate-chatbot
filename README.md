# Chat With Document

### Environment
```
conda create -n chatdoc python=3.11.5
conda activate chatdoc
```

### Installations
```
pip install -r requirements.txt

pip install pre-commit
pre-commit install

./scripts/run-dev.sh
mypy --install-types
```

### Execution
```
./scripts/run-dev.sh
```

### Access
```
(Docs) http://127.0.0.1:8000/api/docs
(Bots) http://127.0.0.1:8000/api/v1/chatdoc
(chatbot) http://127.0.0.1:8000/chatdoc
(chatbot) http://127.0.0.1:8000/realestatebot
(initial questions)
```

### Docker
```
(install) snap install docker
(build)   sudo docker build -t chatdoc .
(run)     sudo docker run -p 0.0.0.0:8000:8000 chatdoc
(list)    sudo docker ps
(stop)    sudo docker stop container-id
```

### API Structure
```
API
│
├── Router: Chatdoc
│   ├── Endpoint I: Upload
│   ├── Endpoint II: Chatbot
```

### Env Instructions
```
- if USE_OPENAI_API="TRUE" then setting OPENAI_API_KEY value is necessary
```

### Instructions
- As new document is attached on a tab, vector database of old document is removed.
- If 2 users interact at the same time, the vector database of latest user is kept only.
- Separate chat history for each user is maintained. A feature can be added that removes chat history of a user either on 'clear history' request or after fixed time.


