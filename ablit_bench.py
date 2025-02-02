import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import platform
import sys
import torch

from barplot import plot_refusal_scores
from datasets import load_dataset
from erisforge.eris_forge import Forge
from erisforge.scorers.refusal_scorer.expression_refusal_scorer import ExpressionRefusalScorer
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run abliteration benchmark')
    parser.add_argument(
        '-n', '--model_names',
        required=True,
        action='extend',
        nargs='+',
        help='HF Names or local paths of models to benchmark.')
    parser.add_argument(
        '-l', '--num_layers',
        default=2,
        type=int,
        help='Number of layers to test when searching for best '
             'layer to modify. Automatically decides which layers '
             'to prioritize based on heuristics.')
    parser.add_argument(
        '-i', '--n_instructions',
        default=20,
        type=int,
        help='Number of instructions to process when searching for '
             'best modification. Higher is more accurate, lower is '
             'faster. Will also be used for performance evaluation.')
    parser.add_argument(
        '-b', '--batch_size',
        default=10,
        type=int,
        help='Instructions processed per batch - '
             'higher is faster, lower uses less memory.')
    return parser.parse_args()

# Download datasets:
def get_harmful_instructions():
    hf_path = 'Undi95/orthogonal-activation-steering-TOXIC'
    dataset = load_dataset(hf_path)
    instructions = [i['goal'] for i in dataset['test']]
    train, test = train_test_split(instructions,
                                   test_size=0.2,
                                   random_state=42)
    return {'train': train, 'test': test}

def get_harmless_instructions():
    hf_path = 'tatsu-lab/alpaca'
    dataset = load_dataset(hf_path)
    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset['train'])):
        if dataset['train'][i]['input'].strip() == '':
            instructions.append(dataset['train'][i]['instruction'])
    train, test = train_test_split(instructions,
                                   test_size=0.2,
                                   random_state=42)
    return {'train': train, 'test': test}


if __name__ == '__main__':
    args = parse_arguments()
    dataset = {'harmful': get_harmful_instructions(),
               'harmless': get_harmless_instructions()}

    refusal_scores_baseline = []
    refusal_scores_intervention = []
    for model_name in args.model_names:
        print("\n\n\n##################\n\n\n"
             f"Running ablation benchmark for model: {model_name}"
              "\n\n\n##################\n\n\n")

        # Initialize erisforge object:
        forge = Forge()

        # Load model and tokenizer:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(forge.device)

        try:
            # TODO: try using model.config["num_layers"]
            tot_number_of_layers = len(model.model.layers)
        except AttributeError:
            print(f"{model_name} is not supported. Its HF "
                  "implementation must have a `model` attribute.",
                  file=sys.stderr)
            continue

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Assess refusal score on base model:
        max_inst = args.n_instructions
        batch_size = args.batch_size
        conversations_standard = []
        subdataset = dataset['harmful']['test'][:max_inst]
        for batch in range(0, max_inst, batch_size):
            conversations_standard.extend(forge.evaluate_base_model(
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=50,
                instructions=subdataset[batch:min(batch + batch_size, max_inst)]
            )
        )

        # Initialize scorer object:
        scorer = ExpressionRefusalScorer()

        # Calculate refusal score for base model:
        refusal_scores = []
        for conv in conversations_standard:
            refusal_scores.append( scorer.score(user_query=conv[0]["content"], model_response=conv[1]["content"]) )
            # print(f'\nUser query:\n{conv[0]["content"]}\n')
            # print(f'Model response:\n{conv[1]["content"]}')
        refusal_score_base = sum(refusal_scores)/len(refusal_scores)
        print(
            '\n'
            f'Model before ablation has {refusal_score_base:.2f}'
            f' refusal score - {sum(refusal_scores):.0f} harmful'
            ' prompts refused over {len(refusal_scores)} prompts.'
            '\n'
        )

        # Select layers:
        min_layer = int( (tot_number_of_layers / 2) - (args.num_layers / 2) )
        max_layer = int( min_layer + args.num_layers )

        # Find direction through layer search and run ablation:
        result_dict = forge.approx_best_objective_behaviour_dir(
            model=model,
            tokenizer=tokenizer,
            scorer=scorer,
            eval_objective_behaviour_instructions=dataset['harmful']['train'][:max_inst],
            eval_antiobjective_instructions=dataset['harmless']['train'][:max_inst],
            min_layer=min_layer,
            max_layer=max_layer,
        )

        # Get the lowest refusal score after ablation (dict is sorted in an ascending order by score):
        final_refusal_score = result_dict[0]['score'] 

        refusal_scores_baseline.append(refusal_score_base)
        refusal_scores_intervention.append(final_refusal_score)
        # Print results:
        print("\n"
              "Refusal score before ablation: ", refusal_score_base)
        print("Refusal score after ablation: ", final_refusal_score)
        print("Refusal Drop Rate: ", refusal_score_base - final_refusal_score)

    plot_refusal_scores(
        args.model_names,
        refusal_scores_baseline,
        refusal_scores_intervention
    )
