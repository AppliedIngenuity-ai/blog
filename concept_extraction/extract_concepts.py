import typer
from lib.prompt_tools import extract_concepts
import os

from dotenv import load_dotenv
load_dotenv() 
ENV_DATA = os.environ # lets load the data from the .env file
finetuned_model = None

if "EXTRACTION_MODEL" in ENV_DATA:
    finetuned_model=ENV_DATA["EXTRACTION_MODEL"]

def main(
        debug: bool = typer.Option(
        help="Enable debugging output",
        default=False
    ),
    text: str = typer.Option(
        help="The text you want to extract concepts from",
        default=None
    ),
    model: str = typer.Option(
        help="The name of the fine-tuned model for concept extraction",
        default=finetuned_model
    )
    ):

    if model is None:
        print("You must specify a fine-tuned model - either on the command-line or in the .env 'EXTRACTION_MODEL'")
        exit(1)

    concepts = extract_concepts(text, model, max_tokens=100, debug=debug, separator="|", stop_string="\n", completion_indicator_string=' ->', num_retries=2)
    print(text + " --> " + ", ".join(concepts))
    return concepts

if __name__ == "__main__":
    typer.run(main)