# ============================================================
# Q3: EFFICIENT SPELLING CORRECTOR
# PART 5: LIVE INTERACTIVE TERMINAL CLI
# ============================================================

import math
import re
import string
import time
from collections import Counter, defaultdict

import nltk
from nltk.corpus import brown


# ============================================================
# 1. LOAD BROWN CORPUS
# ============================================================

print("\nLoading Brown Corpus...")

nltk.download("brown", quiet=True)

brown_words = brown.words()

# Keep only lowercase alphabetic words
words = [
    word.lower()
    for word in brown_words
    if re.fullmatch(r"[a-z]+", word.lower())
]

# ============================================================
# 2. VOCABULARY + UNIGRAM MODEL
# ============================================================

word_freq = Counter(words)

vocabulary = set(word_freq.keys())

total_words = len(words)

unigram_prob = {
    word: count / total_words
    for word, count in word_freq.items()
}


# ============================================================
# 3. BIGRAM MODEL
# ============================================================

bigram_freq = Counter(
    zip(words[:-1], words[1:])
)

bigram_prob = {
    (w1, w2): count / word_freq[w1]
    for (w1, w2), count in bigram_freq.items()
}


# ============================================================
# 4. METHOD A
# Standard Edit Distance 1
# ============================================================

def edit_distance_1(word):

    candidates = set()

    # --------------------------------------------------------
    # Deletions
    # --------------------------------------------------------

    for i in range(len(word)):
        candidates.add(
            word[:i] + word[i + 1:]
        )

    # --------------------------------------------------------
    # Transpositions
    # --------------------------------------------------------

    for i in range(len(word) - 1):
        candidates.add(
            word[:i]
            + word[i + 1]
            + word[i]
            + word[i + 2:]
        )

    # --------------------------------------------------------
    # Replacements
    # --------------------------------------------------------

    for i in range(len(word)):

        for ch in string.ascii_lowercase:

            if ch != word[i]:

                candidates.add(
                    word[:i]
                    + ch
                    + word[i + 1:]
                )

    # --------------------------------------------------------
    # Insertions
    # --------------------------------------------------------

    for i in range(len(word) + 1):

        for ch in string.ascii_lowercase:

            candidates.add(
                word[:i]
                + ch
                + word[i:]
            )

    return candidates


def method_a_candidates(word):

    return edit_distance_1(word).intersection(
        vocabulary
    )


# ============================================================
# 5. METHOD B
# Symmetric Delete
# ============================================================

print("Building Symmetric Delete index...")

delete_index = defaultdict(set)

for word in vocabulary:

    for i in range(len(word)):

        deleted = (
            word[:i]
            + word[i + 1:]
        )

        delete_index[deleted].add(word)

print("Models ready.")


def method_b_candidates(word):

    candidates = set()

    for i in range(len(word)):

        deleted = (
            word[:i]
            + word[i + 1:]
        )

        if deleted in delete_index:

            candidates.update(
                delete_index[deleted]
            )

    return candidates


# ============================================================
# 6. NON-WORD ERROR CORRECTION
# Uses BOTH Method A and Method B
# ============================================================

def correct_non_word(word):

    # Method A
    candidates_a = method_a_candidates(word)

    # Method B
    candidates_b = method_b_candidates(word)

    # Combine both candidate sets
    candidates = candidates_a.union(
        candidates_b
    )

    if not candidates:
        return word

    # Select highest unigram probability
    best_candidate = max(
        candidates,
        key=lambda candidate:
            unigram_prob.get(candidate, 0)
    )

    return best_candidate


# ============================================================
# 7. BIGRAM CONTEXT MODEL
# ============================================================

def bigram_log_probability(w1, w2):

    probability = bigram_prob.get(
        (w1, w2),
        1e-10
    )

    return math.log(probability)


def context_score(
    sentence,
    index,
    candidate
):

    score = 0.0

    # Previous word -> candidate
    if index > 0:

        previous_word = sentence[index - 1]

        score += bigram_log_probability(
            previous_word,
            candidate
        )

    # Candidate -> next word
    if index < len(sentence) - 1:

        next_word = sentence[index + 1]

        score += bigram_log_probability(
            candidate,
            next_word
        )

    return score


# ============================================================
# 8. REAL-WORD ERROR CORRECTION
# ============================================================

def correct_real_word(
    sentence,
    index,
    threshold=2.0
):

    original_word = sentence[index]

    # Generate edit-distance-1 candidates
    candidates = method_a_candidates(
        original_word
    )

    # Always include the original word
    candidates.add(original_word)

    candidate_scores = {}

    for candidate in candidates:

        candidate_scores[candidate] = (
            context_score(
                sentence,
                index,
                candidate
            )
        )

    original_score = candidate_scores[
        original_word
    ]

    best_candidate = max(
        candidate_scores,
        key=candidate_scores.get
    )

    best_score = candidate_scores[
        best_candidate
    ]

    # Only make the correction when the
    # contextual improvement is significant.
    if (
        best_candidate != original_word
        and best_score >= original_score + threshold
    ):
        return best_candidate

    return original_word


# ============================================================
# 9. TOKENIZATION
# ============================================================

def tokenize_text(text):

    return re.findall(
        r"[A-Za-z]+|[^A-Za-z]+",
        text
    )


# ============================================================
# 10. COMPLETE TEXT CORRECTION
# ============================================================

def correct_text(text):

    parts = tokenize_text(text)

    # Extract alphabetic words for language model
    sentence_words = [
        part.lower()
        for part in parts
        if re.fullmatch(
            r"[A-Za-z]+",
            part
        )
    ]

    if not sentence_words:
        return text, []

    corrected_words = sentence_words.copy()

    # --------------------------------------------------------
    # Stage 1: Non-word errors
    # --------------------------------------------------------

    for i, word in enumerate(
        corrected_words
    ):

        if word not in vocabulary:

            corrected_words[i] = (
                correct_non_word(word)
            )

    # --------------------------------------------------------
    # Stage 2: Real-word errors
    # --------------------------------------------------------

    for i, word in enumerate(
        corrected_words
    ):

        if word in vocabulary:

            corrected_words[i] = (
                correct_real_word(
                    corrected_words,
                    i
                )
            )

    # --------------------------------------------------------
    # Reconstruct sentence
    # --------------------------------------------------------

    result = []

    changed_words = []

    word_index = 0

    for part in parts:

        # Alphabetic word
        if re.fullmatch(
            r"[A-Za-z]+",
            part
        ):

            original_word = part
            corrected_word = corrected_words[
                word_index
            ]

            # Check whether this word changed
            if (
                original_word.lower()
                != corrected_word
            ):

                changed_words.append(
                    (
                        original_word,
                        corrected_word
                    )
                )

            # Preserve capitalization
            if original_word.isupper():

                corrected_word = (
                    corrected_word.upper()
                )

            elif (
                len(original_word) > 0
                and original_word[0].isupper()
            ):

                corrected_word = (
                    corrected_word[0].upper()
                    + corrected_word[1:]
                )

            result.append(
                corrected_word
            )

            word_index += 1

        # Punctuation / spaces
        else:

            result.append(part)

    corrected_text = "".join(result)

    return corrected_text, changed_words


# ============================================================
# 11. HIGHLIGHT CHANGED WORDS
# ============================================================

def build_highlighted_output(
    text,
    changed_words
):

    if not changed_words:
        return text

    parts = tokenize_text(text)

    highlighted_parts = []

    change_index = 0

    for part in parts:

        if re.fullmatch(
            r"[A-Za-z]+",
            part
        ):

            if change_index < len(changed_words):

                original_word, corrected_word = (
                    changed_words[change_index]
                )

                # Highlight only if this is the
                # corresponding changed word.
                if part.lower() == corrected_word.lower():

                    highlighted_parts.append(
                        "\033[93m"
                        + part
                        + "\033[0m"
                    )

                    change_index += 1

                else:

                    highlighted_parts.append(
                        part
                    )

            else:

                highlighted_parts.append(
                    part
                )

        else:

            highlighted_parts.append(
                part
            )

    return "".join(highlighted_parts)


# ============================================================
# 12. TERMINAL USER INTERFACE
# ============================================================

print()
print("=" * 65)
print("             NLP SPELL CORRECTION SYSTEM")
print("=" * 65)

print("Brown Corpus vocabulary:", len(vocabulary), "words")

print()
print("Enter a sentence and press Enter.")
print("Changed words are highlighted in yellow.")
print("Type 'exit' to close the application.")
print("=" * 65)


while True:

    try:

        user_input = input(
            "\nEnter sentence: "
        ).strip()

    except KeyboardInterrupt:

        print("\n\nApplication terminated.")
        break

    except EOFError:

        print("\n\nApplication terminated.")
        break

    # --------------------------------------------------------
    # Exit command
    # --------------------------------------------------------

    if user_input.lower() == "exit":

        print(
            "\nExiting spell correction system..."
        )

        break

    # --------------------------------------------------------
    # Empty input
    # --------------------------------------------------------

    if not user_input:

        print(
            "Please enter a sentence."
        )

        continue

    # --------------------------------------------------------
    # Run correction + measure latency
    # --------------------------------------------------------

    start_time = time.perf_counter()

    corrected_text, changed_words = (
        correct_text(user_input)
    )

    end_time = time.perf_counter()

    latency_ms = (
        end_time - start_time
    ) * 1000

    # --------------------------------------------------------
    # Highlight changes
    # --------------------------------------------------------

    highlighted_text = (
        build_highlighted_output(
            corrected_text,
            changed_words
        )
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\nOriginal:")
    print(user_input)

    print("\nCorrected:")
    print(highlighted_text)

    if changed_words:

        print("\nChanges:")

        for original, corrected in changed_words:

            print(
                f"  {original} -> {corrected}"
            )

    else:

        print("\nChanges:")
        print("  No spelling corrections made.")

    print(
        f"\nLatency: {latency_ms:.3f} ms"
    )