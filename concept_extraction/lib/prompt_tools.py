### This is from AppliedIngenuity.ai Substack/Blog
### https://github.com/AppliedIngenuity-ai/blog/tree/main

import openai
import random
import re
import copy
import os

from dotenv import load_dotenv
load_dotenv() 
ENV_DATA = os.environ # lets load the data from the .env file
openai.api_key = ENV_DATA["OPENAI_API_KEY"] # set up the OpenAI key using the key stored in .env file

DEFAULT_MODEL = "gpt-3.5-turbo" #"gpt-4" # If you don't have access to GPT-4 or prefer the cheaper alternative "gpt-3.5-turbo"
DEFAULT_COMPLETION_MODEL = "text-davinci-003"

def make_prompt_from_template(prompt_template, replacement_dictionary):
    """This function will take a string (prompt_template) which contains instances of
    {{X}}  where X is a string corresponding to a key in replacement_dictionary
    
    For example:
    
    A prompt template might look like:
    -----
    my_prompt_template = ...You are an AI, please generate {{num}} queries about {{subject}}.

    EXAMPLES:
    {{examples}}
    -----

    Calling with:
    my_repl_dict = {"num" : "5", "subject" : "math", "examples" : examples_string}
    make_prompt_from_template(my_prompt_template, my_repl_dict)

    will replace {{num}} with "5", {{suject}} with "math" and {{examples}} with the value of examples_string
    """
    
    prompt = prompt_template
    for key, value in replacement_dictionary.items():
        replace_match = "{{" + key + "}}"
        prompt = prompt.replace(replace_match, str(value))
    return prompt


def get_chat_response(prompt, temperature=0.0, max_tokens=1500, debug=False, model=DEFAULT_MODEL, num_retries=2):
    num_tries = 0
    done = False
    while num_tries <= num_retries and done == False:
        num_tries += 1
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            res = response["choices"][0]["message"]["content"]
            if debug:
                print(response["usage"])
            return res
        except Exception as e:
            print("Failed try: ", num_tries, " Execption: ", e)
    return None

def get_completion_response(prompt, temperature=0.0, max_tokens=2000, debug=False, model=DEFAULT_COMPLETION_MODEL, num_retries=2, stop_string=None):
    num_tries = 0
    done = False
    while num_tries <= num_retries and done == False:
        num_tries += 1
        try:
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_string
            )

            res = response['choices'][0]["text"]

            if debug:
                print(response["usage"])
            return res
        except Exception as e:
            print("Failed try: ", num_tries, " Execption: ", e)
    return None

def get_completion_response_with_probs(prompt, temperature=0.0, max_tokens=2000, debug=False, 
                                       model=DEFAULT_COMPLETION_MODEL, num_retries=2, stop_string=None,
                                      logprobs=2):
    num_tries = 0
    done = False
    while num_tries <= num_retries and done == False:
        num_tries += 1
        try:
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_string,
                logprobs=logprobs
            )
            
            if debug:
                print(response["usage"])
            return response
        except Exception as e:
            print("Failed try: ", num_tries, " Execption: ", e)
    return None

def generate_misspelling(text, error_rate=0.02):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    misspelling = []
    error_types = ['substitution', 'insertion', 'deletion', 'swap']
    
    for char in text:
        if random.random() < error_rate:
            error_type = random.choice(error_types)
            
            if error_type == 'substitution':
                new_char = random.choice(alphabet)
                misspelling.append(new_char)
            elif error_type == 'insertion':
                new_char = random.choice(alphabet)
                misspelling.append(char)
                misspelling.append(new_char)
            elif error_type == 'deletion':
                continue
            elif error_type == 'swap':
                if len(misspelling) > 0:
                    prev_char = misspelling.pop()
                    misspelling.append(char)
                    misspelling.append(prev_char)
        else:
            misspelling.append(char)
    
    return ''.join(misspelling)


split_newlines = re.compile(r'\n')
def extract_examples_from_response(response, *match_tokens):
    """This function will take a chat response which contains multiple generated examples.
    Each generated example is assumed to be a list item (or items) separated by newlines.

    For example:
    A single output sample might be, and others are newline separated. (there could be more than one newline between sets)
    
    Q: hello this is a query about doggs
    C: dogs

    Output is an array of dictionaries where the keys are the separators (match_tokens) i.e.
    [{"Q":"hello this is a query about doggs", "C":"dogs"}, .....]
    """
    
    # we can split on newlines? then split (in-order for each of the parts???)
    split_output = split_newlines.split(response.strip()) # strip removes leading/trailing spaces
    
    # now we need to search for the matches
    # We can do a state machine - we can skip "invalid lines" if "blank" we can be okay
    # The match_tokens is an array of string sequences i.e. "Q:", "C:"
    # We can skip extras optionally i.e. maybe there is a debug
    
    results = []
    cur_index = 0 # what we are looking for
    num_match_tokens = len(match_tokens)
    tok_map = dict([(x[0], x[1]) for x in enumerate(match_tokens)])
 
    cur_res = dict()
    for row in split_output:
        # we check if it begins with the next expected token - if not we report an error?
        matched = False
        cur_match_token=tok_map[cur_index]
        if row.strip() == "":
            continue
            
        if row.startswith(cur_match_token):
            # we are good!!!
            cur_res[cur_match_token]=row.replace(cur_match_token,"").strip()
            matched = True
        cur_index += 1
        if cur_index == num_match_tokens:
            results.append(copy.deepcopy(cur_res))
            cur_res = dict()
            cur_index = 0
        if matched == False:
            print("warn didn't get expected {} for {}".format(cur_match_token, row))
    return results

def extract_concepts(text, fine_tuned_model, max_tokens=100, debug=False, separator="|", stop_string="\n",
                     completion_indicator_string=' ->', num_retries=2, min_log_prob=-0.9):
    """ This function takes an input text, then sends it to a fine_tuned completion model designed to do concept extraction.
    The output is separated by some separator pipe (|) for example and ends with the stop_string such as newline.
    For example, an input text might be: "the yankees are the best basebal team" and the completion output to the prompt:
    might be: "New York Yankees|baseball team". To reduce generation of bad concepts, we added a filter on the log_prob
    of the first token for any concept 
    """

    prompt = text + completion_indicator_string
    resp = get_completion_response_with_probs(prompt, temperature=0.0, max_tokens=max_tokens, debug=debug, 
                                              model=fine_tuned_model, num_retries=num_retries, stop_string=stop_string)

    token_list = resp['choices'][0]['logprobs']['tokens']
    log_probs = resp['choices'][0]['logprobs']['token_logprobs']
    
    if debug:
        print([(token_list[i], log_probs[i]) for i in range(0, len(token_list))])
    
    # now to generate the output by scanning and filtering
    extracted_concepts = []
    cur_text = ""
    first_token_log_prob = -9999
    for i in range(0, len(token_list)):
        # if the token is "|" then we decide to save or skip
        cur_token = token_list[i]
        cur_token_log_prob = log_probs[i]
        if debug:
            print("checking: {}, {}".format(cur_token_log_prob, cur_token))
        if cur_text == "" and cur_token != "|":
            cur_text = cur_token.lstrip() # we strip leading spaces (really only applies to the very first token)
            first_token_log_prob = cur_token_log_prob
        elif cur_token == "|": # this denotes there is another token
            if first_token_log_prob >= min_log_prob:
                extracted_concepts.append(cur_text)
            else:
                if debug:
                    print("skipping {} since {} < {}".format(cur_text,first_token_log_prob, min_log_prob))
            cur_text = ""    
            continue
        else:
            cur_text += cur_token
        
    # check the last
    if cur_text.strip() != "":
        if first_token_log_prob >= min_log_prob:
                extracted_concepts.append(cur_text)
        else:
            if debug:
                print("skipping {} since {} < {}".format(cur_text,first_token_log_prob, min_log_prob))
    return extracted_concepts