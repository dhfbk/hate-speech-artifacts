import emoji


EPSILON = 1e-16

EMOJI_TOKEN = "[EMOJI]"
EMOJIS_TOKENS = list(emoji.UNICODE_EMOJI['en'].keys())

LEXART_TEMPLATE_SEC1_TITLE = "i) Top lexical artifacts."
LEXART_TEMPLATE_SEC2_TITLE = "ii) Class definitions."
LEXART_TEMPLATE_SEC3_TITLE = "iii) Methods and resources."

LEXART_TEMPLATE_SEC1_CONTENT = [
    "We present the top",
    "most informative tokens for the",
    "class along with their scores"
]
LEXART_TEMPLATE_SEC2_CONTENT = [
    "[INSERT HERE THE DEFINITION FOR THE",
    "CLASS.]"
]
LEXART_TEMPLATE_SEC3_CONTENT = [
    "In order to compute the correlation between tokens to the",
    "class we employ [INSERT HERE THE METHOD USED FOR COMPUTING LEXICAL ARTIFACTS].",
    "[INSERT HERE THE DETAILS ON PREPROCESSING AND DUPLICATES HANDLING.]",
    "[INSERT HERE THE LINKS TO RELATED RESOURCES (e.g., FULL LIST OF LEXICAL ARTIFACTS).]"
]

LATEX_IMPORTS = "\n\n% Notes: for correct table formatting, include the following:\n% \\usepackage{{booktabs}}"

NEWLINE = "\n"
OPEN_TEXTSC = "\\textsc{"
CLOSE_TEXTSC = "}~~~"

LATEX_TABLE_PART1 = """
\\begin{table}
    \\centering
    \\begin{tabular}{rlr}
        \\toprule
        \\emph{Rank} & Token & Score \\\\
        \\midrule
"""
LATEX_TABLE_PART2 = """
        \\bottomrule
    \\end{tabular}
"""
