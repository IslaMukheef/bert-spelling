import pandas as pd

def parse_birkbeck_dat(filepath):
    misspelled_words = []
    correct_words = []

    with open(filepath, "r") as file:
        current_correct_word = None
        
        for line in file:
            # Strip whitespace and check if the line is not empty
            line = line.strip()
            if not line:
                continue

            # If the line starts with $, it's a correct word
            if line.startswith("$"):
                current_correct_word = line.lstrip("$")
            else:
                # Otherwise, it's a misspelled word related to the last correct word
                if current_correct_word:
                    misspelled_words.append(line)
                    correct_words.append(current_correct_word)

    # Create a DataFrame to hold the data
    df = pd.DataFrame({
        "misspelled": misspelled_words,
        "correct": correct_words
    })
    
    # Save to CSV for easy access later
    df.to_csv("birkbeck_misspellings.csv", index=False)
    print("Data saved to birkbeck_misspellings.csv")

# Run the function on your .dat file
parse_birkbeck_dat("birkbeck.dat")
