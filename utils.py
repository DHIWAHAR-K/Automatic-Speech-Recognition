#utils.py
import os
import matplotlib.pyplot as plt

def plot_and_save_history(history, graph_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    graph_path = os.path.join(graph_dir, 'loss_graph.png')
    plt.savefig(graph_path)
    plt.close()