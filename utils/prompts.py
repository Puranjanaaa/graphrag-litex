"""
Prompt templates for GraphRAG.
"""
from typing import List

ENTITY_EXTRACTION_PROMPT = """
---Goal---
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

---Steps---
1. Identify all entities. For each identified entity, extract the following information:
- entity name: Name of the entity, capitalized
- entity type: One of the following types: [{entity_types}]
- entity description: Comprehensive description of the entity's attributes and activities

Format each entity as ("entity"|<entity name>|<entity type>|<entity description>)

2. From the entities identified in step 1, identify all pairs of (source entity, target entity) that are *clearly related* to each other
For each pair of related entities, extract the following information:
- source entity: name of the source entity, as identified in step 1
- target entity: name of the target entity, as identified in step 1
- relationship description: explanation as to why you think the source entity and the target entity are related to each other
- relationship strength: a numeric score between 0 and 1 indicating strength of the relationship between the source entity and target entity

Format each relationship as ("relationship"|<source entity>|<target entity>|<relationship description>|<relationship strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use ** as the list delimiter.

4. When finished, output END_OF_EXTRACTION
---Examples---
Entity types: ORGANIZATION,PERSON
Input:
The Fed is scheduled to meet on Tuesday and Wednesday, with the central bank planning to release its latest policy decision on Wednesday at 2:00 p.m. ET, followed by a press conference where Fed Chair Jerome Powell will take questions. Investors expect the Federal Open Market Committee to hold its benchmark interest rate steady in a range of 5.25%-5.5%.

Output:
("entity"|FED|ORGANIZATION|The Fed is the Federal Reserve, which is setting interest rates on Tuesday and Wednesday)**("entity"|JEROME POWELL|PERSON|Jerome Powell is the chair of the Federal Reserve)**("entity"|FEDERAL OPEN MARKET COMMITTEE|ORGANIZATION|The Federal Reserve committee makes key decisions about interest rates and the growth of the United States money supply)**("relationship"|JEROME POWELL|FED|Jerome Powell is the Chair of the Federal Reserve and will answer questions at a press conference|0.9)
END_OF_EXTRACTION

---Real Data---
Entity types: {entity_types}
Input:
{text_chunk}

Output:
"""

ENTITY_REFLECTION_PROMPT = """
Given the following list of entities and relationships that were extracted from a text document, please identify any entities that were missed in the extraction. The goal is to ensure that all relevant entities and relationships are captured.

Original text document:
{text_chunk}

Extracted entities and relationships:
{extracted_entities}

Were all relevant entities, of types [{entity_types}], successfully extracted from the document? Answer with YES or NO.
"""

ENTITY_REFLECTION_CONTINUATION_PROMPT = """
MANY entities were missed in the last extraction. Please identify the missed entities and any relationships between them and previously extracted entities. Follow the same format as before:

For entities: ("entity"|<entity name>|<entity type>|<entity description>)
For relationships: ("relationship"|<source entity>|<target entity>|<relationship description>|<relationship strength>)

Original text document:
{text_chunk}

Previously extracted entities and relationships:
{extracted_entities}

Output the missed entities and relationships:
"""

CLAIM_EXTRACTION_PROMPT = """
---Goal---
Given a text document, extract factual claims related to the entities mentioned in the text.

---Steps---
1. From the provided text, identify all factual claims related to each entity. A factual claim is a statement that explicitly presents a verifiable fact about an entity.

2. For each factual claim:
- Ensure it is a simple, self-contained statement that conveys a single fact
- Ensure it is directly supported by the text and not an inference or opinion
- Connect it to the relevant entity or entities mentioned

3. Format each claim as: ("claim"|<claim content>|<entity1>,<entity2>,...) where:
- claim content: The factual claim as a concise statement
- entity1, entity2, etc.: The names of the entities this claim is about, separated by commas

4. Return a list of all extracted claims, separated by **

5. When finished, output END_OF_CLAIMS

---Example---
Input:
Apple Inc. reported record revenue of $123.9 billion in Q1 2022, with iPhone sales accounting for over 50% of the total. CEO Tim Cook announced that supply chain issues were improving. The company's shares rose 5% following the announcement.

Output:
("claim"|Apple Inc. reported record revenue of $123.9 billion in Q1 2022|APPLE INC.)**("claim"|iPhone sales accounted for over 50% of Apple's total revenue in Q1 2022|APPLE INC.,IPHONE)**("claim"|Tim Cook is the CEO of Apple Inc.|TIM COOK,APPLE INC.)**("claim"|Tim Cook announced that supply chain issues were improving|TIM COOK,SUPPLY CHAIN)**("claim"|Apple's shares rose 5% following the revenue announcement|APPLE INC.)
END_OF_CLAIMS

---Real Data---
Text document:
{text_chunk}

Known entities:
{entity_names}

Output:
"""

COMMUNITY_SUMMARY_PROMPT = """
---Role---
You are an AI assistant that helps a human analyst to perform general information discovery. Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

---Goal---
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

---Report Structure---
The report should include the following sections:
- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community. IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
{{
"title": <report title>,
"summary": <executive summary>,
"rating": <impact severity rating>,
"rating explanation": <rating explanation>,
"findings": [
  {{
    "summary":<insight 1 summary>,
    "explanation": <insight 1 explanation>
  }},
  {{
    "summary":<insight 2 summary>,
    "explanation": <insight 2 explanation>
  }}
]
}}

---Grounding Rules---
Points supported by data should list their data references as follows:
"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where 1, 5, 7, 23, 2, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.

---Real Data---
Use the following text for your answer. Do not make anything up in your answer.

Input:
{input_text}

Output:
"""

COMMUNITY_ANSWER_PROMPT = """
Your task is to analyze the following report and return a structured JSON object.

{report}

Return a JSON object with the following schema:
{{
  "helpfulness": <integer score 0-100>,
  "topics": [<string>, ...],
  "answer": <detailed markdown answer>
}}
Only return the JSON. Do not include extra text.
"""

GLOBAL_ANSWER_PROMPT = """
---Role---
You are a helpful assistant responding to questions about a comprehensive dataset.

---Goal---
Generate a response that answers the following question:

{question}

Synthesize all relevant information from the community answers provided below. These answers contain different perspectives on the question from analysis of different parts of the dataset.

If you don't know the answer based on the provided information, just say so. Do not make anything up.

The response should be comprehensive, well-structured, and formatted in markdown. Include sections, bullet points, or other formatting as appropriate to make the information clear and accessible.

Preserve all data references from the community answers using the format:
"This is a supported claim [Data: Reports (2, 7, 34, 46, 64, +more)]."

Create an answer that gives the most complete picture possible based on all the community answers. Focus on presenting diverse perspectives and comprehensive coverage of the topic.

---Community Answers---
{community_answers}

Generate a markdown-formatted response that answers the question comprehensively:
"""

class PromptTemplates:
    """
    Provides templating functionality for LLM prompts.
    """

    @staticmethod
    def format_community_answer_prompt(question: str, report_data: str) -> str:
        return COMMUNITY_ANSWER_PROMPT.format(
            question=question,
            report=report_data
        )

    @staticmethod
    def format_global_answer_prompt(question: str, community_answers: str) -> str:
        return GLOBAL_ANSWER_PROMPT.format(
            question=question,
            community_answers=community_answers
        )

    @staticmethod
    def format_community_summary_prompt(input_text: str) -> str:
        return f"""You are an expert analyst. Analyze the following structured information from a knowledge graph community and produce a JSON summary with the following fields:

    - title (string): A short, informative title for the community.
    - summary (string): A high-level summary of what this community is about.
    - rating (number between 0-10): How important or central is this community?
    - rating explanation (string): Justify your rating briefly.
    - findings (list of objects): Each with:
        - summary (string): Short finding
        - explanation (string): Explanation of the finding

    Respond only in JSON format.

    Community Data:
    -----------------------
    {input_text}
    """
