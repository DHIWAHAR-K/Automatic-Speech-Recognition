#text_process.py
import torch

class TextProcess:
    def __init__(self):
        # Define character to index mapping
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}  # Mapping from character to index
        self.index_map = {}  # Mapping from index to character

        # Populate the character to index and index to character mappings
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch

        # Map index 1 to space character
        self.index_map[1] = ' '

    def text_to_int_sequence(self, text):
        """
        Convert text to a sequence of integers using the character map.
        
        Args:
            text (str): Input text.
            
        Returns:
            list: List of integers representing the text.
        """
        # Convert each character in the text to its corresponding integer index
        return [self.char_map['<SPACE>'] if c == ' ' else self.char_map[c] for c in text]

    def int_to_text_sequence(self, labels):
        """
        Convert a sequence of integers back to text using the index map.
        
        Args:
            labels (list): List of integers.
            
        Returns:
            str: The resulting text string.
        """
        # Convert each integer in the list to its corresponding character
        return ''.join([self.index_map[i] for i in labels]).replace('<SPACE>', ' ')

# Instantiate the TextProcess class
textprocess = TextProcess()

def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    """
    Decodes the output of the model using greedy decoding.
    
    Args:
        output (torch.Tensor): The output tensor from the model.
        labels (torch.Tensor): The ground truth labels.
        label_lengths (torch.Tensor): The lengths of the labels.
        blank_label (int): The index of the blank label.
        collapse_repeated (bool): Whether to collapse repeated characters.
        
    Returns:
        tuple: Decoded sequences and target sequences.
    """
    arg_maxes = torch.argmax(output, dim=2)  # Get the index of the max log-probability
    decodes = []
    targets = []

    # Iterate over the batch
    for i, args in enumerate(arg_maxes):
        decode = []
        # Convert the ground truth labels to text
        targets.append(textprocess.int_to_text_sequence(labels[i][:label_lengths[i]].tolist()))
        
        # Convert the model output to text
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(textprocess.int_to_text_sequence(decode))
    return decodes, targets