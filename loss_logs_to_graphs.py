import re
import matplotlib.pyplot as plt

def parse_log(log_content):
    pattern = r"Epoch \d+/\d+ \| Batch (\d+)/\d+ \| Avg Loss: ([\d.]+)"
    matches = re.findall(pattern, log_content)
    batches = [int(match[0]) for match in matches]
    losses = [float(match[1]) for match in matches]
    return batches, losses

def plot_loss(batches, losses, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(batches, losses)
    plt.title(title)
    plt.xlabel('Batch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def loss_to_graph(lr):
    with open(f'logs/trained_on_llama_data_{lr}_modloss.log', 'r') as file:
        llama_log_content = file.read()

    with open(f'logs/trained_on_claude_data_{lr}_modloss.log', 'r') as file:
        claude_log_content = file.read()

    llama_batches, llama_losses = parse_log(llama_log_content)
    claude_batches, claude_losses = parse_log(claude_log_content)

    plot_loss(llama_batches, llama_losses, 'Average Loss per Batch (LLaMA)', f'graphs/llama_loss_graph_{lr}_modloss.png')
    plot_loss(claude_batches, claude_losses, 'Average Loss per Batch (Claude)', f'graphs/claude_loss_graph_{lr}_modloss.png')

    print(f"Graphs have been saved as 'llama_loss_graph_{lr}_modloss.png' and 'claude_loss_graph_{lr}_modloss.png'")

    plt.figure(figsize=(12, 6))
    plt.plot(llama_batches, llama_losses, label='LLaMA')
    plt.plot(claude_batches, claude_losses, label='Claude')
    plt.title('Average Loss per Batch (LLaMA vs Claude)')
    plt.xlabel('Batch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'graphs/combined_loss_graph_{lr}_modloss.png')
    plt.close()

    print(f"Combined graph has been saved as 'combined_loss_graph_{lr}_modloss.png'")

if __name__ == "__main__":
    loss_to_graph(1e-6)