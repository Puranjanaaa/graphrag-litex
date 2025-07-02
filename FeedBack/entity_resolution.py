from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Using the same Embedding model in the file entity_resolver.py 
# Model all-MiniLM-L6-v2 
# Threshold 0.85

'''
.../graphrag-litex/extraction/entity_resolver.py

def __init__(self, threshold: float = 0.85):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold

'''

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(entity1, entity2):
    
    embeddings = model.encode([entity1, entity2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

def semantic_entity_resolution(entities, threshold=0.85):
    processed = set()
    
    for i, entity1 in enumerate(entities):
        if entity1 in processed:
            continue
            
        similar_group = [entity1]
        processed.add(entity1)
        
        for j, entity2 in enumerate(entities[i+1:], i+1):
            if entity2 in processed:
                continue
                
            similarity = compute_similarity(entity1, entity2)
            print(f"Similarity between '{entity1}' and '{entity2}': {similarity:.3f}")
            
            if similarity >= threshold:
                similar_group.append(entity2)
                processed.add(entity2)
                print(f"INCORRECTLY MERGED: '{entity2}' merged into '{entity1}' group")
        
        

def demonstrate_problems():

    # Problem 1: AWS ecosystem entities
    aws_entities = [
        "Amazon web services Load Balancer", 
        "Amazon Load Balancer Configuration"
    ]

    location_entities = [
        "New York" , 
        "New York City"
    ]
    
    
    semantic_entity_resolution(aws_entities, threshold= 0.85)
    semantic_entity_resolution(location_entities , threshold=0.85)
    
   

if __name__ == "__main__":
    demonstrate_problems()
    