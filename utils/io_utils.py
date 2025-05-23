import os

def read_text_files(directory: str) -> dict:
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith((".txt", ".md", ".py", ".java", ".js", ".html", ".css")):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                documents[filename] = f.read()
    return documents
