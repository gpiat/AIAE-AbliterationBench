import sys
sys.path.append('/Users/damifass/Desktop')

import platform
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from erisforge.eris_forge import Forge
from erisforge.scorers import ExpressionRefusalScorer

import matplotlib.pyplot as plt
import numpy as np

from datasets import load_dataset
from sklearn.model_selection import train_test_split

def print_memory_usage(prefix=""):
    """Prints memory usage information (GPU or CPU)."""

    if torch.cuda.is_available():  # Check for CUDA (NVIDIA GPU)
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        free = torch.cuda.mem_get_info(0)[0]

        print(f"{prefix}GPU Allocated: {allocated / 1024**3:.2f} GB")
        print(f"{prefix}GPU Cached: {cached / 1024**3:.2f} GB")
        print(f"{prefix}GPU Free: {free / 1024**3:.2f} GB")
        print(f"{prefix}GPU Total: {(allocated + free) / 1024**3:.2f} GB")

    elif platform.system() == "Darwin":  # Check for macOS (Apple Silicon or Intel)
        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            used_memory = mem_info.rss / (1024 ** 3)  # Resident Set Size
            available_memory = psutil.virtual_memory().available / (1024 ** 3)

            print(f"{prefix}System Used Memory: {used_memory:.2f} GB")
            print(f"{prefix}System Available Memory: {available_memory:.2f} GB")
            print(f"{prefix}System Total Memory: {(used_memory + available_memory):.2f} GB")

        except ImportError:
            print(f"{prefix}psutil not found. Cannot display system memory usage. Install with: pip install psutil")
        except Exception as e:
            print(f"{prefix}Error getting system memory info: {e}")

    else:
        print(f"{prefix}Neither CUDA nor macOS detected. Cannot determine memory usage.")


# Instructions:
# code lifted from abliterator
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

# Initialize ErisForge 
forge = Forge()

# datasets to be used for caching and testing, split by harmful/harmless
dataset = [get_harmful_instructions(), get_harmless_instructions()] # Format: [harmful, harmless]

max_inst = 20

print_memory_usage()

# Load a model and tokenizer
model_name = "Qwen/Qwen-1.5-0.5B-Chat"

# Load the model with specific settings for device compatibility
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency if supported
).to(forge.device)  # Move model to the device set in forge (e.g., GPU if available)

# Initialize the tokenizer with the model's configuration
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

scorer = ExpressionRefusalScorer()

# Assess refusal score on base model:
batch_size = 10 
conversations_standard = []
for batch in range(0, len(dataset[0][1][:max_inst]), batch_size):
    conversations_standard.extend(forge.evaluate_base_model(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    instructions=dataset[0][1][:max_inst][batch:min(batch + batch_size, len(dataset[0][1][:max_inst]))]
    )
)
    

for conv in conversations_standard:
    # print(f'\nUser query:\n{conv[0]["content"]}\n')
    print(f'Model response:\n{conv[1]["content"]}')
    break

refusal_scores = []
for conv in conversations_standard:
    refusal_scores.append( scorer.score(user_query=conv[0]["content"], model_response=conv[1]["content"]) )
    # print(f'\nUser query:\n{conv[0]["content"]}\n')
    # print(f'Model response:\n{conv[1]["content"]}')
refusal_score_base = sum(refusal_scores)/len(refusal_scores)

print(f'''\nModel before ablation has {refusal_score_base:.2f} refusal score - {sum(refusal_scores):.0f} harmful prompts refused over {len(refusal_scores)} prompts.\n''')


print_memory_usage()

number_of_layers = len(model.model.layers)
min_layer = int(len(model.model.layers) * 0.7)
max_layer = int(len(model.model.layers) * 0.75)

result_dict = forge.find_approximate_best_objective_behaviour_direction(
    model=model,
    tokenizer=tokenizer,
    scorer=scorer,
    eval_objective_behaviour_instructions=dataset[0][1][:max_inst],
    eval_antiobjective_instructions=dataset[1][1][:max_inst],
    min_layer=min_layer,
    max_layer=max_layer,
)
result_dict
# First score is lowest:
final_refusal_score = result_dict[0]['score'] 

print("\nRefusal score before ablation: ", refusal_score_base)
print("Refusal score after ablation: ", final_refusal_score)
print("\nRefusal Drop Rate: ", refusal_score_base - final_refusal_score)

# Refusal score: the lower the better
# print_memory_usage()

# torch.mps.empty_cache()

def plot_refusal_scores(model_names, refusal_score_baseline, refusal_score_intervention, safety_score_baseline=None, safety_score_intervention=None):
    """
    Plots refusal and safety scores for different models.

    Args:
        model_names: A list of strings representing the names of the models.
        refusal_score_baseline: A list of floats representing the refusal scores for the baseline model.
        refusal_score_intervention: A list of floats representing the refusal scores for the intervention model.
        safety_score_baseline (optional): A list of floats representing the safety scores for the baseline model.
        safety_score_intervention (optional): A list of floats representing the safety scores for the intervention model.
    """

    num_models = len(model_names)
    x = np.arange(num_models)  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, refusal_score_baseline, width, label='Refusal score (baseline)', color='skyblue', edgecolor='black')
    rects2 = ax.bar(x, refusal_score_intervention, width, label='Refusal score (intervention)', color='steelblue', edgecolor='black')

    # Plot safety scores if provided
    if safety_score_baseline and safety_score_intervention:
        rects3 = ax.bar(x + width, safety_score_baseline, width, label='Safety score (baseline)', color='orange', hatch='//', edgecolor='black')  # hatch for distinction
        rects4 = ax.bar(x + 2 * width, safety_score_intervention, width, label='Safety score (intervention)', color='darkorange', hatch='//', edgecolor='black')  # hatch for distinction

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_title('Refusal scores over harmful instructions')
    ax.set_xticks(x + width/2 if safety_score_baseline else x)  # adjust xticks based on safety plot presence
    ax.set_xticklabels(model_names, rotation=45, ha='right')  # Rotate labels for better readability
    ax.legend(fontsize='small')  # smaller legend font

    # Add value labels on top of bars for better readability
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize='8')  # smaller font size

    autolabel(rects1)
    autolabel(rects2)
    if safety_score_baseline and safety_score_intervention:
        autolabel(rects3)
        autolabel(rects4)

    fig.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.ylim(0, 1.05)  # Set y-axis limit slightly above 1 for better visualization
    plt.show()


model_names = [model_name]
refusal_score_baseline = [refusal_score_base]
refusal_score_intervention = [final_refusal_score]

plot_refusal_scores(model_names, refusal_score_baseline, refusal_score_intervention)
