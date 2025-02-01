import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add value labels on top of bars for better readability
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize='8')  # smaller font size

def plot_refusal_scores(model_names,
                        refusal_scores_base,
                        refusal_scores_post,
                        safety_scores_base=None,
                        safety_scores_post=None):
    """
    Plots refusal and safety scores for different models.

    Args:
        model_names: A list of strings representing the names of the models.
        refusal_scores_base: A list of floats representing the refusal scores for the baseline model.
        refusal_scores_post: A list of floats representing the refusal scores for the intervention model.
        safety_scores_base (optional): A list of floats representing the safety scores for the baseline model.
        safety_scores_post (optional): A list of floats representing the safety scores for the intervention model.
    """

    num_models = len(model_names)
    x = np.arange(num_models)  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width,
                    refusal_scores_base,
                    width,
                    label='Refusal score (baseline)',
                    color='skyblue', edgecolor='black')
    rects2 = ax.bar(x,
                    refusal_scores_post,
                    width,
                    label='Refusal score (intervention)',
                    color='steelblue', edgecolor='black')

    # Plot safety scores if provided
    if safety_scores_base and safety_scores_post:
        rects3 = ax.bar(x + width, safety_scores_base, width, label='Safety score (baseline)', color='orange', hatch='//', edgecolor='black')  # hatch for distinction
        rects4 = ax.bar(x + 2 * width, safety_scores_post, width, label='Safety score (intervention)', color='darkorange', hatch='//', edgecolor='black')  # hatch for distinction

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_title('Refusal scores over harmful instructions')
    ax.set_xticks(x + width/2 if safety_scores_base else x)  # adjust xticks based on safety plot presence
    ax.set_xticklabels(model_names, rotation=45, ha='right')  # Rotate labels for better readability
    ax.legend(fontsize='small')  # smaller legend font


    autolabel(rects1, ax)
    autolabel(rects2, ax)
    if safety_scores_base and safety_scores_post:
        autolabel(rects3, ax)
        autolabel(rects4, ax)

    fig.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.ylim(0, 1.05)  # Set y-axis limit slightly above 1 for better visualization
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Plot abliteration benchmark results')
    parser.add_argument('-n', '--model_names',
                        nargs='+', required=True)
    parser.add_argument('-rb', '--ref_scores_base',
                        nargs='+', required=True, type=float)
    parser.add_argument('-rp', '--ref_scores_post',
                        nargs='+', required=True, type=float)
    parser.add_argument('-sb', '--safe_scores_base',
                        nargs='+', default=None, type=float)
    parser.add_argument('-sp', '--safe_scores_post',
                        nargs='+', default=None, type=float)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    plot_refusal_scores(args.model_names, 
                        args.ref_scores_base, 
                        args.ref_scores_post,
                        args.safe_scores_base,
                        args.safe_scores_post)
