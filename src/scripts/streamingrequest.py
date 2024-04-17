# python3 /src/scripts/streamingrequest.py
import requests
import sys

# Sending the query in the get request parameter

while True:
    query = input("\nQuery : ")
    if query == "quit":
        break

    # sends user query
    url = f"http://127.0.0.1:8000/stream/?query={query}"

    # recieve response
    with requests.get(url, stream=True) as r:
        sys.stdout.write("Response : ")
        # until streaming, keep printing
        for chunk in r:
            sys.stdout.write(chunk.decode("utf-8") + " ")
            sys.stdout.flush()
        sys.stdout.write("\n")
