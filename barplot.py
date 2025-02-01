import argparse
import matplotlib.pyplot as plt
import numpy as np

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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run abliteration benchmark')
    parser.add_argument('-n', '--model_names')
    parser.add_argument('-rb', '--ref_score_baseline')
    parser.add_argument('-ri', '--ref_score_intervention')
    return parser.parse_args()

def main(model_names, refusal_score_baseline, refusal_score_intervention):
    plot_refusal_scores(model_names, refusal_score_baseline, refusal_score_intervention)

if __name__ == "__main__":
    args = parse_arguments()
    model_names = args.model_names
    refusal_score_baseline = args.ref_score_baseline
    refusal_score_intervention = args.ref_score_intervention
    main()