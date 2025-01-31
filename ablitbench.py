# -*- coding: utf-8 -*-
# !pip install "git+https://github.com/Tsadoq/ErisForge.git#egg=erisforge" -r https://raw.githubusercontent.com/Tsadoq/ErisForge/main/requirements.txt

## Install from pip
# !pip install erisforge
!pip install datasets

import argparse
import random
import torch

from datasets import load_dataset
from erisforge import Forge
from erisforge.scorers import ExpressionRefusalScorer
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# code lifted from FailSpy/abliterator
def get_harmful_instructions():
    hf_path = 'Undi95/orthogonal-activation-steering-TOXIC'
    dataset = load_dataset(hf_path)
    instructions = [i['goal'] for i in dataset['test']]

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test

def get_harmless_instructions():
    hf_path = 'tatsu-lab/alpaca'
    dataset = load_dataset(hf_path)
    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset['train'])):
        if dataset['train'][i]['input'].strip() == '':
            instructions.append(dataset['train'][i]['instruction'])

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test

def parse_args():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')

    parser.add_argument('model_name', 
                        default="Qwen/Qwen-1.5-0.5B-Chat")
    parser.add_argument('max_inst', default=100)
    parser.add_argument('batch_size', default=10)
    parser.add_argument('-s', '--search', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    # Instantiating the model
    args = parse_args()

    # Load a model and tokenizer
    model_name = args.model_name

    # Initialize ErisForge and configure the scorer
    forge = Forge()
    scorer = ExpressionRefusalScorer()

    # Load the model with specific settings for device compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency if supported
    ).to(forge.device)  # Move model to the device set in forge (e.g., GPU if available)
    # Initialize the tokenizer with the model's configuration
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # datasets to be used for caching and testing, split by harmful/harmless
    dataset = [get_harmful_instructions(), get_harmless_instructions()] # Format: [(harmful train, harmful test), (harmless train, harmless test)]
    # TODO: This format kind of sucks, we should reformat it into a dict or smth

    # print([(len(a), len(b)) for a, b in dataset])

    # objective behavior instructions = harmful
    # anti-objective behavior instructions = harmless
    forge.load_instructions(
        objective_behaviour_instructions=dataset[0][0],
        anti_behaviour_instructions=dataset[1][0]
    )

    ### Tokenizing Instructions
    # Max instructions to process
    max_inst = args.max_inst
    batch_size = args.batch_size
    # Tokenize the instructions for objective and anti-objective behaviors
    tokenized_instructions = forge.tokenize_instructions(
        # TODO: Tokenizer requires chat templates. Can we do away with that requirement?
        tokenizer=tokenizer,
        max_n_antiobjective_instruction=max_inst,
        max_n_objective_behaviour_instruction=max_inst,
    )


    ## Find Direction for Objective Behavior Transformation

    ### Computing outputs for instructions
    # They will be used to find the actual direction in the residual
    # stream + eval the model before abliteration

    model_responses = {"obj_beh": [], "anti_obj": []}
    for batch in range(0, max_inst, batch_size):
        temp = forge.compute_output(
                model=model,
                objective_behaviour_tokenized_instructions=tokenized_instructions["objective_behaviour_tokens"][batch:min(batch + batch_size, max_inst)],
                anti_behaviour_tokenized_instructions=tokenized_instructions["antiobjective_tokens"][batch:min(batch + batch_size, max_inst)],
            )
        model_responses["obj_beh"].extend(temp["obj_beh"])
        model_responses["anti_obj"].extend(temp["anti_obj"])
    
    forge.free_memory([tokenized_instructions])


    ## Testing model pre-abliteration

    refusals_pre_ablit = []
    for conversation in zip(dataset[1][0][:max_inst], 
                            model_responses["anti_obj"]):
        refusals_pre_ablit.append(
            scorer.score(user_query=conversation[0],
                         model_response=conversation[1])
        )

    # TODO: output some more meaningful metrics here
    print("Mean refusal Score before Abliteration:", sum(refusals_pre_ablit)/len(refusals_pre_ablit))


    """### Computing heuristic-based direction
    Uses a layer near the middle. Won't work as well as testing the directions to choose the best one, but is significantly cheaper to run
    """

    layer = int(len(model.model.layers) * 0.65)  # TODO: make into param

    if not args.search:
        refusal_dir = forge.compute_objective_behaviour_direction(
            model=model,
            objective_behaviour_outputs=model_responses["obj_beh"], # OBJective BEHavior (harmful)
            antiobjective_outputs=model_responses["anti_obj"], # ANTI OBJective (harmless)
            layer=layer,  # Use a specific layer to apply the transformation, in this case a layer kind of in the middle is chosen because it's a good starting point
        )

    refusal_dir.shape

    """### Automatically finding best refusal direction
    Runs out of memory on Qwen 1.5 0.5B if `max_inst > 25`
    """

    min_layer = int(len(model.model.layers) * 0.4)
    max_layer = int(len(model.model.layers) * 0.8)
    if args.search:
        try:
            refusal_dir = forge.find_approximate_best_objective_behaviour_direction(
                model=model,
                tokenizer=tokenizer,
                scorer=scorer,
                eval_objective_behaviour_instructions=dataset[0][1][:max_inst],
                eval_antiobjective_instructions=dataset[1][1][:max_inst],
                min_layer=min_layer,
                max_layer=max_layer,
            )
            print("Best direction computed successfully.")
        except Exception as e:
            print("An error occurred during computation:", e)
            print("This may be due to memory constraints or a memory leak.")

    """## Modifying and the model and running it"""

    batch_size = 10 # TODO: Make into script parameter
    instructions=random.sample(population=dataset[0][1], k=100)

    conversations = []
    for batch in range(0, len(instructions), batch_size):
        conversations.extend(
            forge.run_forged_model(
                model=model,
                objective_behaviour_dir=refusal_dir,
                tokenizer=tokenizer,
                instructions=instructions[batch:min(batch + batch_size, len(instructions))],
                max_new_tokens=100,
                stream=False,
            )
        )

    refusals_post_ablit = []
    for conversation in conversations:
        # TODO: this exception is only there for testing/debug purposes. It may have issues if the roles can vary in names (e.g. "AI" instead of "assistant").
        if conversation[0]["role"] != "user" or conversation[1]["role"] != "assistant":
            print(f"Starting role: {conversation[0]['role']}\nFollowing role: {conversation[1]['role']}")
            raise ValueError("Conversation should start with a user prompt and be followed by an assistant response.")
        refusals_post_ablit.append(
            scorer.score(user_query=conversation[0]["content"],
                         model_response=conversation[1]["content"])
        )

    # TODO: output some more meaningful metrics here
    print("Mean refusal Score after Abliteration:", sum(refusals_post_ablit)/len(refusals_post_ablit))

    """Could be useful to look at the output conversations for testing purposes. Leaving it commented so we can just "execute all cells" without polluting the outputs."""

    for conversation in conversations:
        print("=" * 20)
        for round in conversation:
            print(f'{round["role"]}: {round["content"]}')

    forge.free_memory([conversations, model_responses])  # Free memory after testing

    # """## Exporting model"""

    # forge.save_model(
    #     model=model,
    #     tokenizer=tokenizer,
    #     behaviour_dir=refusal_dir,
    #     output_model_name=f"{model_name}_abliterated",  # Name for the saved model
    #     to_hub=False,  # Set to True to push the model to the HuggingFace Hub
    #     # Using None, ErisForge will try to guess the model architecture.
    #     # This could be replaced by a variable and specified manually.
    #     # The list of architectures is the keys in the dict defined by this JSON file:
    #     # https://github.com/Tsadoq/ErisForge/blob/main/erisforge/assets/llm_models.json
    #     model_architecture=None,
    # )
