import openai
from openai import OpenAI
#Creation of OPENAI assisstant which will be used only once
client = OpenAI()
assisstant = client.beta.assistants.create(
            name="DERMASCAN",
            instructions="""You're a board-certified dermatologist AI assistant. 
        Combine medical knowledge with the user's skin lesion analysis from our custom model.
        - Use provided diagnostic data when available
        - Never diagnose without image analysis
        - Cite sources for treatment recommendations""",
        model='gpt-3.5-turbo'
        )

ASSISTANT_ID = assisstant.id

