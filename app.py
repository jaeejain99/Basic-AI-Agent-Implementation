from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

import os
os.environ["HF_TOKEN"] = "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"
# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def my_custom_tool(arg1:str, arg2:int)-> str: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that generates an image based on a text prompt.
    Args:
        arg1: text description of the image
        arg2: image resolution (not used in this tool, but kept for compatibility)
    """
    try:
        # Generate an image using the image generation tool
        generated_image = image_generation_tool(arg1)

        # If the tool returns an image URL or object, process it accordingly
        return f"Generated image for '{arg1}': {generated_image}"
    
    except Exception as e:
        return f"Error generating image: {str(e)}"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, my_custom_tool, get_current_time_in_timezone], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()