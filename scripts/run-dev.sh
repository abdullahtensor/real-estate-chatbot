#!/bin/bash

export $(cat .env | xargs)
echo "Running server on port $PORT"

MYPYPATH=./src mypy src/app/main.py

if [ $? -ne 0 ]; then
    echo "Mypy had some errors please fix before proceeding..."
    exit $?
fi


black src/
pylint -j4 src/
cd src/

uvicorn app.main:app --reload

cd ..