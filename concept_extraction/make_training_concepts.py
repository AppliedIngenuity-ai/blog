import typer
from lib.prompt_tools import make_prompt_from_template, get_chat_response, extract_examples_from_response, generate_misspelling
import sys
import os
sys.path.append(os.getcwd())
import json
import random
from pathlib import Path # Note this is from Python 3.4+ only

def construct_fine_tune_entry(example, error_rate=0, completion_end="->", stop_token="\n"):
    """The format for a retrain as described in: https://platform.openai.com/docs/guides/fine-tuning
    is a file of one-line JSON as follows:
    {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
    """

    # Our "example" is a dictionary with keys: "T:" and "C:"
    if "T:" not in example or "C:" not in example:
        print("warning, missing required T: or C: in example: {}".format(json.dumps(example)))
    base_text = example["T:"]
    if error_rate > 0:
        base_text = generate_misspelling(base_text, error_rate=error_rate)

    res = {
                "prompt" : base_text + " ->",
                "completion" : " " + example["C:"].strip() + "\n"
            }
    
    return res

def main(
        debug: bool = typer.Option(
        help="Enable debugging output",
        default=False,
    ),
    config_file: str = typer.Option(
        help="Path to a JSON config file with the reqired options",
        default="./configs/make_training_concepts_config.json"
    ),
    output_file: str = typer.Option(
        help="Where do we save the generated output to?",
        default="./data/generated_concept_extraction_examples.jsonl"
    ),
    num_docs: int = typer.Option(
        help="How many document chunks to include in the prompt?",
        default=7
    ),
    num_to_generate: int = typer.Option(
        help="How many examples do we want to ask it to generate per iteration?",
        default=20
    ),
    num_iterations: int = typer.Option(
        help="How many iterations to run",
        default=1
    ),
    overwrite: bool = typer.Option(
        help="If set, then the system will overwrite the output file if present - cannot set both overwrite and append",
        default=False
    ),
    append: bool = typer.Option(
        help="If set, then the system will append the output file if present - cannot set both overwrite and append",
        default=False
    )
):

    if debug:
        print("Debug is enabled")
        print("Config file is {}".format(config_file))

    # Lets open the config file
    with open(config_file, "r") as CFG:
        config_json = json.load(CFG)
        
    if debug:
        print("using config: {}".format(json.dumps(config_json, indent=2)))

    # lets load the prompt template!
    with open(config_json['prompt_template_file']) as PTF:
        prompt_template = PTF.read()

    # lets load the data (this could easily be pulled from a DB instead)
    with open(config_json['source_documents_file']) as DF:
        all_docs = DF.readlines()

    # Some parameter setups etc...
    # Since the OpenAI tools will auto-delete dupes, we don't need to worry about it

    # Lets check if the output file already exists:
    out_file = Path(output_file)
    if out_file.is_file():
        if overwrite:
            print("Looks like your output file {} already exists, erasing since overwrite is set".format(out_file))
            os.remove(out_file)
        elif append:
            print("Looks like your output file {} already exists, since append is set, will append to the end of the file".format(out_file))
        else:
            print("Looks like your output file {} already exists, please move or remove and try again, or set append or overwrite".format(out_file))
            exit(1)

    num_generated = 0
    if append:
        write_or_append = "a"
    else:
        write_or_append = "w"
    
    with open(out_file, write_or_append) as OF:

        # Now to do iterations and generaiton!
        for i in range(0, num_iterations):
            print("Iteration: {}".format(i+1))

            # Lets select the random documents
            selected_docs = [json.loads(x)['text'] for x in random.sample(all_docs, num_docs)]
            selected_docs_string = "\n\n".join(selected_docs)

            # Lets construct the prompt from the template!
            repl_dict = {
                "num" : str(num_to_generate),
                "text_blocks" : selected_docs_string
            }
            prompt=make_prompt_from_template(prompt_template, repl_dict)

            if debug:
                print("Prompt is:\n\n{}".format(prompt))

            # lets get the results!!!
            response = get_chat_response(prompt, debug=debug)
            
            if debug:
                print("RAW RESPONSE:\n{}\n\n".format(response))
            #if debug:
            #    print("\nraw response:\n{}".format(response))
            # Lets process this and get the outputted examples
            examples = extract_examples_from_response(response, "T:", "C:")
            
            print("Got {} examples, saving...".format(len(examples)))
            
            if debug:
                print("generated examples: \n{}".format(examples))

            # Now lets add misspellings to the text and save
            for example in examples:
                res = construct_fine_tune_entry(example, error_rate=config_json['error_rate'])
                OF.write(json.dumps(res) + "\n")

            num_generated += len(examples)

    print("done generating {} examples for file: {}".format(num_generated, out_file))
    print("You can combine multiple of the generated files together into one big file. Once you have enough total examples you can perform a fine-tuning.")
    print("https://platform.openai.com/docs/guides/fine-tuning")

if __name__ == "__main__":
    typer.run(main)