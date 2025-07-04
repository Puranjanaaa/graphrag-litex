from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Optional, Literal
import json


# Below we have defined two pydantic classes. These will be used to validate the LLM response
# When using pydantic we can validate each field separately and also constrain each field to a specific set of accepted outputs


class TopicModel(BaseModel):
    name: str = Field(..., min_length=1, description="Topic name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    category: Optional[str] = Field(None, description="Topic category")

class LLMAnalysisResponse(BaseModel):
    answer: str = Field(..., min_length=1, description="The main answer")
    topics: List[TopicModel] = Field(default_factory=list, description="Extracted topics")
    used_relationships: List[str] = Field(default_factory=list, description="Relationships used")
    used_chunks: List[int] = Field(default_factory=list, description="Chunk IDs used")
    sentiment: Literal["positive", "negative", "neutral"] = Field("neutral", description="Overall sentiment")
    
    @field_validator('used_chunks')
    @classmethod
    def validate_chunk_ids(cls, v):
        for chunk_id in v:
            if chunk_id < 0:
                raise ValueError(f"Chunk ID must be positive, got {chunk_id}")
        return v

# This function will be used to run the validation check , we pass in the raw LLM output string to the function. 
# For simplicity we would only accept LLM final response output strings but we can extend this to accept all the LLM outputs 


def validate_llm_response(llm_output: str) -> LLMAnalysisResponse:
    """
    Validate LLM JSON output using Pydantic model
    
    Args:
        llm_output: Raw string output from LLM
    
    Returns:
        Validated LLMAnalysisResponse instance
    
    Raises:
        ValidationError: If output doesn't match schema
        json.JSONDecodeError: If output isn't valid JSON
    """
    try:
        
        parsed_json = json.loads(llm_output)
        return LLMAnalysisResponse(**parsed_json)
            
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {e}")
    except ValidationError as e:
        raise ValueError(f"Validation failed: {e}")



# here we have a valid LLM output 

valid_llm_output = """
{
    "answer": "The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th centuries.",
    "topics": [
        {"name": "Renaissance", "confidence": 0.95, "category": "History"},
        {"name": "European Culture", "confidence": 0.87, "category": "Culture"}
    ],
    "used_relationships": ["historical_period", "cultural_movement"],
    "used_chunks": [1, 3, 7],
    "sentiment": "neutral"
}
"""

# This is a invalid output , notice that we are missing a few important fields 
# used_relationship is missing 
# sentiment is missing 
# also note that used chunk is a negative value i.e -1 which is wrong - note how we have a separate validate function for that 
# If we pass -1 to the client we will have issues as its not a valid ID 

invalid_llm_output = """
{
    "topics": [
        {"name": "Renaissance", "confidence": 1.5}
    ],
    "used_chunks": [-1, 3]
}
"""



def main():

    
    print("1. Testing valid LLM output:")
    try:
        validated_response = validate_llm_response(valid_llm_output)
        print("✅ Validation successful!")
        print(f"Answer: {validated_response.answer}")
        print(f"Topics found: {len(validated_response.topics)}")
        print(f"First topic: {validated_response.topics[0].name} (confidence: {validated_response.topics[0].confidence})")
        print(f"Used chunks: {validated_response.used_chunks}")
        
        # Now we can safely access the data and pass it into other sections of the code. 
        for topic in validated_response.topics:
            print(f"  - {topic.name}: {topic.confidence:.2f}")
            
    except ValueError as e:
        print(f"❌ Validation failed: {e}") # This wont trigger for the valid example - just added this to complete the try catch block 
    
   
    
 
    print("2. Testing invalid LLM output:")
    try:
        validated_response = validate_llm_response(invalid_llm_output)
        print("Validation successful!") # This wont trigger since the input is invalid - agian adding this to complete the try catch block 
    except ValueError as e:
        print(f"Validation failed (as expected): {e}")

if __name__ == "__main__":
    main()