#!/bin/bash

export $(cat .env | xargs)
echo "Running server on port $PORT"

cd src/

uvicorn app.main:app --host 0.0.0.0 --port $PORT
#uvicorn app.main:app --reload

cd ..
