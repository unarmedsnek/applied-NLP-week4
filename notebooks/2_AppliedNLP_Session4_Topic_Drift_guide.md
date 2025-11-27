## Character-Centered Topic Drift Across the Epics — Notebook Guide

This document explains the notebook `2_AppliedNLP_Session4_Topic_Drift.ipynb` in **painstaking detail**.  
It is written for:

- **Beginners** who want a step-by-step explanation of what the code does and why.  
- **Data scientists** who care about the modeling choices, limitations, and how to interpret the results.  
- **Humanities / literature readers** who want to understand what these numbers say about the *Iliad* and *Odyssey*.

The notebook studies **“topic drift”** along **character-centered narratives** in Homer’s *Iliad* and *Odyssey* using modern NLP tools.

---

## 1. High-Level Overview

### 1.1. What the notebook does, in plain language

- Loads English translations of Homer’s **Iliad** and **Odyssey** (Butler translation).
- Splits each epic into **paragraphs**, grouped by **book** (Book I, II, …).
- Detects when key **characters** (Achilles, Odysseus, Hector, Athena, Zeus, etc.) appear in each paragraph.
- Turns each paragraph into a **vector** (embedding) using a transformer model (`all-MiniLM-L6-v2`).
- Measures how **similar** or **different** consecutive appearances of a character are in meaning (using **cosine similarity** between embeddings).
- Summarizes these similarities as a notion of **“topic drift”**:
  - High similarity → **continuity** (the story stays on the same kind of topic).
  - Low similarity → **drift** (the story has moved into a very different situation).
- Compares:
  - Characters **within** an epic (e.g. Achilles vs Hector in the *Iliad*).
  - Characters **across** epics (e.g. Odysseus in the *Iliad* vs Odysseus in the *Odyssey*).
- Cross-checks results with:
  - A second embedding model (`all-MiniLM-L12-v2`).
  - A **lexical** measure of similarity (word overlap).
- Provides:
  - Plots of drift over time.
  - Example paragraphs with very **high** and very **low** similarity.
  - A concise **summary table** and “Key Findings & Limitations” section.

### 1.2. Important caveats (what this is **not**)

This notebook is **not** a proof of deep literary theses such as “Achilles is definitively more volatile than Hector.”  
It is:

- A **quantitative, exploratory analysis** that highlights **patterns**, suggests hypotheses, and points you to interesting parts of the text.
- Dependent on:
  - **Regex-based character detection** (no full coreference resolution).
  - **Partial episode tags** (only major books get labels like “Cyclops Polyphemus”).
  - **English translation** (Butler), not the original Greek.

You should treat the results as:

- **Signals**: where there might be strong continuity or big shifts.
- **Guides for close reading**: where to look in the text.
- **Inputs for NLP design**: e.g. chunking long texts around high-drift boundaries.

---

## 2. How to Run and Use the Notebook

### 2.1. Environment and dependencies

The project has a `requirements.txt` with the key libraries:

- `numpy`, `matplotlib` — numeric computing and plotting.
- `sentence-transformers`, `transformers`, `torch` — embedding models.
- `scipy` — statistics (`pearsonr`).

Install dependencies (from project root):

```bash
pip install -r requirements.txt
```

Open `2_AppliedNLP_Session4_Topic_Drift.ipynb` in Jupyter, VS Code, or another notebook environment.

### 2.2. Run order

The notebook is designed so that you can do a **full “Run All”** safely:

1. Cells 0–12: **Legacy global analysis** (combined *Iliad*+*Odyssey*).  
   - Labeled as **Appendix (Legacy Global Analysis – Optional)**.  
   - Kept for teaching only; not used in the final results.

2. Cell 13: **Methodological Caveats** (markdown).  
   - Read this before interpreting any numbers.

3. Cells 14–22: **Refined per-epic analysis**, segment-level drift, and extra visualizations.  
   - These are the main, serious analysis cells.

4. Cells 27–31: **Interpretation, summary table, and key findings**.  
   - These summarize what the numbers say.

5. Cell 30 (final): **Final cleanup** (optional).  
   - Only run this if you want to free memory **after** all analysis and summaries have run.

> **Important**: There is an older mid-notebook cleanup cell (now a no-op `pass`); it does nothing and is safe to ignore. The only real cleanup is the *last* cell.

### 2.3. What to look at if you’re in a hurry

If you just want the **results** and **story**:

- Read:
  - Cell 13: **Methodological Caveats**.
  - Cell 14: **Refined Per-Epic Topic Drift Analysis** introduction.
  - The plots produced by:
    - The per-epic drift (cell 18).
    - The segment-level summaries (cell 22).
    - The extra visualizations (cells 24–26).
  - Cell 28–29: **Summary table** and **Key Findings & Limitations**.

If you want to **recompute everything**:

- Just **Run All**.

---

## 3. Legacy Global Analysis (Appendix)

Cells 3–12 perform a **simpler, earlier version** of the analysis:

- They:
  - Combine the *Iliad* and *Odyssey* into one giant text (`combined_text`).
  - Split into paragraphs using blank lines.
  - Encode all paragraphs with `all-MiniLM-L6-v2` (`para_embeds`).
  - Detect characters with a **very simple substring lookup** (e.g. “achilles” in the text).
  - Compute drift as cosine similarity between consecutive paragraphs where the character appears, regardless of epic or book.
  - Plot per-character drift and basic statistics.

This section is useful to see the **basic idea**:

> “Take character paragraphs → embed them → compare neighbors with cosine similarity.”

But it has serious limitations:

- Merges *Iliad* and *Odyssey* — you can’t distinguish trends by epic.
- No book/episode awareness.
- Very naive character detection (no epithets, no pronouns).

The refined analysis **fixes** these issues and is what you should use for interpretation.

---

## 4. Refined Per-Epic Pipeline — Step by Step

The main analysis starts at the **“Refined Per-Epic Topic Drift Analysis”** heading (cell 14).

### 4.1. Book- and paragraph-aware segmentation

**Purpose:** Split each epic into paragraphs organized by **book**, and attach useful metadata.

Key parts:

- `split_books_and_paragraphs(text, min_words=10)`:
  - Normalizes newlines.
  - Scans through lines looking for patterns like `Book I`, `Book II`, etc. using `BOOK_PATTERN`.
  - Groups lines by book.
  - Within each book, splits into paragraphs on blank lines (`\n\s*\n+`).
  - Cleans whitespace and filters out very short paragraphs (`min_words`).
  - Builds a list of dicts:
    - `text` — the paragraph text.
    - `book` — Roman numeral of the book (e.g. `"I"`, `"IX"`), or `None` for preface.
    - `para_in_book` — paragraph index within the book.
    - `global_idx` — paragraph index within the epic.

Then:

- `iliad_paragraphs_meta = split_books_and_paragraphs(iliad_text)`  
- `odyssey_paragraphs_meta = split_books_and_paragraphs(odyssey_text)`

**Why this way?**

- We want to know where we are in the epic: **Book X, paragraph Y**.
- Later, we will:
  - Match low-similarity points to **episode labels**.
  - Compare within vs across episodes.

**For a beginner:**  
Think of this as indexing every paragraph with a **chapter number** and a **position in that chapter**, so we can later say “this big change happens right around *Book IX: Cyclops*”.

### 4.2. Episode labels

We define two dictionaries:

- `ILIAD_EPISODES` — e.g. `"I": "Quarrel of Achilles and Agamemnon"`, `"XVI": "Death of Patroclus"`.
- `ODYSSEY_EPISODES` — e.g. `"IX": "Cyclops Polyphemus"`, `"XI": "Nekyia (Underworld)"`, etc.

For each paragraph metadata dict (`meta`), we add:

- `meta["episode"] = ILIAD_EPISODES.get(book)` (or ODYSSEY equivalent).

**Why?**

- This lets us later attach **human-readable labels** to drift points:
  - “This big drop in similarity occurs at ‘Death of Patroclus’”.
- It also allows us to distinguish **within-episode** vs **across-episode** transitions.

**Limitations:**

- Only **major books** are labeled. Many paragraphs have `episode=None`.

### 4.3. Character detection with regex and a simple coreference heuristic

**Goal:** Determine which **tracked characters** appear in each paragraph.

#### 4.3.1. Patterns

We define:

- `CHAR_PATTERNS` — a dict mapping character names to a list of regex patterns, e.g.:
  - `"achilles": [r"\bachilles\b", r"\bson of peleus\b"]`
  - `"odysseus": [r"\bodysseus\b", r"\bulysses\b", r"\bson of laertes\b"]`
  - `"zeus": [r"\bzeus\b", r"\bjove\b"]`
  - `"hera": [r"\bhera\b", r"\bjuno\b"]`

We compile them to `COMPILED_PATTERNS` for efficiency.

**Why regex?**

- Simple substring search misses epithets like “son of Peleus”.
- Regex with `\b` ensures we only match **whole words**, not substrings inside other words.

#### 4.3.2. Explicit character match

- `detect_characters(text)`:
  - For each character, check if any of its patterns matches the paragraph text.
  - Collects all characters explicitly mentioned.

For each paragraph in `*_paragraphs_meta`, we call `detect_characters(meta["text"])` and store the result in `para_chars` (initial detection).

#### 4.3.3. Minimal pronoun-based coreference

We add a tiny heuristic:

- Define `PRONOUNS = {"he", "him", "his", "she", "her"}`.
- `paragraph_has_pronoun(text)` returns `True` if any of these pronouns appears.

Then in `build_character_indices`:

1. We first run `detect_characters` for all paragraphs and store results in `para_chars`.
2. Then we iterate again, maintaining `last_single_char`:
   - If a paragraph has **exactly one** character in `para_chars[idx]`, we set `last_single_char` to that character.
   - If it has **zero** characters and contains a pronoun (`paragraph_has_pronoun` is `True`) **and** `last_single_char` is not `None`, we **inherit** that character:
     - `para_chars[idx].append(last_single_char)`.
   - If it has multiple characters, we reset `last_single_char = None`.
3. Finally, we build `char_indices` from the augmented `para_chars`.

**Intuition in human terms:**

- If a paragraph talks explicitly about “Odysseus” and then the next paragraph only says “he … he … he”, it’s probably still about Odysseus.
- This heuristic captures that case, but **only** when exactly one character is clearly referenced just before.

**For data scientists:**

- This is a cheap, one-step coreference approximation that:
  - Extends unambiguous character runs.
  - Avoids messy multi-character pronoun attribution.
- It’s intentionally conservative but **improves recall** compared to pure regex.

**Outputs:**

- `iliad_para_chars`, `iliad_char_indices`.
- `odyssey_para_chars`, `odyssey_char_indices`.

Each `*_char_indices[char]` is a list of paragraph indices where that character appears.

### 4.4. Per-epic paragraph embeddings

We prepare text:

- `iliad_paragraph_texts = [m["text"] for m in iliad_paragraphs_meta]`
- `odyssey_paragraph_texts = [m["text"] for m in odyssey_paragraphs_meta]`

Then:

- `model = SentenceTransformer("all-MiniLM-L6-v2")` (created in the legacy section but reused here).
- `iliad_embeds = model.encode(iliad_paragraph_texts, show_progress_bar=True)`  
- `odyssey_embeds = model.encode(odyssey_paragraph_texts, show_progress_bar=True)`

**Meaning:**

- Each paragraph becomes a **384-dimensional vector** that captures its semantic content in a way that makes similar paragraphs close in vector space.

**Why this model?**

- `all-MiniLM-L6-v2` is:
  - Small and fast.
  - Well-known, general-purpose sentence embedding model.
- Trade-off:
  - It’s not specialized for archaic epic, but it’s a reasonable baseline for English narrative text.

### 4.5. Context-aware paragraph-level drift

**Goal:** Measure how the semantic content changes between nearby **paragraphs where a character appears**.

#### 4.5.1. Compute drift records

`compute_topic_drift_records(indices, embeddings, paragraphs_meta)`:

- Input:
  - `indices`: list of paragraph indices for a character (from `*_char_indices`).
  - `embeddings`: `iliad_embeds` or `odyssey_embeds`.
  - `paragraphs_meta`: metadata for the corresponding epic.
- Process:
  - Loop through `indices` in order, taking consecutive pairs `(i, j)`.
  - Skip if `abs(j - i) > MAX_GAP` (we set `MAX_GAP = 5`):
    - This ensures the paragraphs are **close in the original narrative**.
  - Skip if either paragraph has `book is None` (front matter).
  - Compute cosine similarity:

    \[
    \text{sim} = \frac{\langle v_i, v_j \rangle}{\|v_i\| \cdot \|v_j\|}
    \]

  - Save a record `{"sim": sim, "i": i, "j": j}`.

For each character and each epic, we store:

- `char_drift_iliad[char]` — array of similarities.
- `char_pairs_iliad[char]` — list of `(sim, i, j)` records.
- Similarly for `*_odyssey`.

**Interpretation:**

- Each similarity value says:  
  “How **semantically close** are two nearby paragraphs where this character appears?”

Higher → more continuity; lower → bigger topic changes.

#### 4.5.2. Drift plots with episode annotations

`plot_character_drift_with_episodes(char, epic="iliad", top_n=3)`:

- Chooses:
  - `sims` and `recs` from `char_drift_iliad/odyssey` and `char_pairs_...`.
  - `meta` from the epic’s paragraph metadata.
- Plots a line of cosine similarity values over the sequence of **character pairs**.
- Identifies the `top_n` **lowest similarity** points (largest jumps) and annotates them with:
  - `book`.
  - `episode` label if available, otherwise “Book X”.

This gives you a visual sense of:

- Where a character’s narrative experiences **sharp topic shifts**.
- How these shifts align with **known episodes**.

### 4.6. Statistical comparisons (randomization test)

`compare_mean_drift(a, b, n_boot=5000)`:

- Takes two arrays of drift values (e.g. Achilles vs Hector in Iliad).
- Computes the observed mean difference `obs_diff = mean(a) - mean(b)`.
- Performs a randomization test:
  - Concatenate `a` and `b`.
  - Repeatedly shuffle and split into pseudo-a / pseudo-b of the same sizes.
  - Compute the mean difference for each shuffle.
  - Estimate the p-value as the fraction of shuffled diffs at least as large in magnitude as `obs_diff`.

Used examples:

- Achilles vs Hector in Iliad.
- Odysseus in Odyssey vs Odysseus in Iliad.

**Plain language:**

- This tells you whether the observed difference in average similarity is likely to be due to **chance** if you ignore which values belong to which group.

**Key result:**

- Achilles vs Hector: no significant difference (p ≈ 0.7).
- Odysseus Odyssey vs Iliad: Odyssey drift is significantly higher (lower mean similarity), with p ≈ 0.000.

### 4.7. Robustness check with an alternate embedding model

**Cell 20** loads `all-MiniLM-L12-v2` as `alt_model` and recomputes:

- `iliad_embeds_alt`, `odyssey_embeds_alt`.
- Drift for Odysseus only (`idxs_i_ody`, `idxs_o_ody`).

Then it prints:

- Iliad mean / std / N with the alt model.
- Odyssey mean / std / N with the alt model.
- Randomization test p-value for Odyssey - Iliad.

**Result:**

- Odyssey still has **lower mean similarity** than Iliad.
- p-value remains **very small (~0.001)**.

**For a data scientist:**

- This shows that the **direction** of the Odysseus effect is not an artifact of one embedding model: it persists across two related models.

### 4.8. Example paragraph pairs

`show_example_pairs(char, epic="iliad", k=1, max_chars=320)`:

- For a given character and epic:
  - Sorts drift similarities ascending (most volatile) and descending (most stable).
  - Prints:
    - For `k` of the highest and `k` of the lowest similarity transitions:
      - Books, paragraph indices, episode labels.
      - First ~320 characters of each paragraph’s text.

This is crucial for **interpretation in human terms**:

- You can read:
  - What “high similarity” transitions look like: usually small shifts within a continuous scene.
  - What “low similarity” transitions look like: often scene boundaries, episode transitions, or big tonal changes.

For beginners:

- It’s like the code is saying:  
  “Here are two parts of the story where Achilles is doing **almost the same thing** or something **completely different**. Read them and judge for yourself.”

### 4.9. Segment-level, multi-metric drift (embeddings + lexical)

**Why segments?**

- Paragraphs are editorial and can:
  - Split or combine smaller narrative moves.
  - Mix multiple characters.
- We instead build **segments** where:
  - A single tracked character appears.
  - No other tracked character appears (if `solo_only=True`).
  - The segment is a run of consecutive paragraphs satisfying that.

#### 4.9.1. Segment construction

`build_character_segments(paragraphs_meta, para_chars, solo_only=True, min_len=1)`:

- For each character:
  - Iterate through paragraphs.
  - If paragraph mentions that character (and solo if specified), add its index to `current`.
  - If a paragraph breaks the run, finalize a segment with:
    - `para_indices` — list of paragraph indices.
    - `text` — concatenated text.
    - `book`, `episode` — dominant book/episode among included paragraphs.
    - `mode` — `"speech"`, `"narration"`, or `"mixed"`, based on whether paragraphs use quotes.

Outputs:

- `iliad_segments_by_char`, `odyssey_segments_by_char`.

#### 4.9.2. Segment embeddings and lexical sets

- `compute_segment_embeddings(segments_by_char, embeddings)`:
  - For each segment, average its constituent paragraph embeddings.
- `tokenize_content_words(text)`:
  - Tokenize, lowercase, remove English stopwords and short tokens.
  - Returns a set of content words.
- `jaccard_similarity(a, b)`:
  - Lexical overlap measure: |A∩B| / |A∪B|.

#### 4.9.3. Segment-level drift

`compute_segment_drift(segments_by_char, seg_embeds_by_char, lag=1)`:

- For each character and epic:
  - Compare consecutive segments (segment `i` vs `i+1`).
  - Compute:
    - `sim_embed`: cosine similarity between segment embeddings.
    - `sim_lex`: Jaccard between content-word sets.
    - `mode_pair`: string like `"speech→speech"`, `"speech→narration"`, etc.
  - Store as a record.

Outputs:

- `iliad_segment_drift`, `odyssey_segment_drift`.

#### 4.9.4. Summaries and mode-pair counts

`summarize_segment_drift(char)`:

- For each epic:
  - Computes:
    - mean, std, N of `sim_embed`.
    - mean, std, N of `sim_lex`.
  - Prints a breakdown of `mode_pair` counts:
    - How often we see speech→speech, narration→speech, etc.

This gives:

- A more robust sense of **continuity** at a slightly higher granularity than paragraphs.
- Insights into whether big shifts tend to happen at **speech↔narration boundaries**.

### 4.10. Additional visualizations

#### 4.10.1. Embedding vs lexical similarity correlation

`plot_segment_drift_correlation(char, epic)`:

- Scatter plot:
  - x-axis: embedding similarity.
  - y-axis: lexical similarity.
- Prints Pearson correlation `r` and p-value.

If `r` is reasonably positive:

- It suggests that embedding-based and word-overlap-based measures are aligned.
- This supports the idea that we’re measuring something real, not pure model noise.

#### 4.10.2. Cross-epic segment continuity for shared characters

`Cross-Epic Segment Continuity for Shared Characters` bar chart:

- For each character that has segments in both epics:
  - Plot two bars:
    - Iliad mean segment-level similarity.
    - Odyssey mean segment-level similarity.

This quickly shows whether a character is:

- More **semantically stable** in one epic or the other.

#### 4.10.3. Within vs across episode drift

`plot_within_vs_across_episode(char, epic)`:

- Uses paragraph-level drift and `episode` labels.
- Creates two groups:
  - Pairs where both paragraphs share the same episode label → **within-episode**.
  - Pairs where labels differ → **across-episode**.
- Plots a boxplot comparing the similarity distributions.

If the across-episode similarities are systematically **lower**:

- It shows that the drift measure respects **episode boundaries**, as we intuitively expect.

---

## 5. Summary Table and Key Findings

### 5.1. Compact drift summary table

The notebook builds a table for **key characters** (`Achilles`, `Odysseus`, `Hector`, `Athena`, `Zeus`) with rows:

- `epic` (Iliad / Odyssey)
- `level`:
  - `"paragraph"` — paragraph-level drift.
  - `"segment_embed"` — segment-level embedding drift.
  - `"segment_lex"` — segment-level lexical drift.
- `mean_sim`, `std_sim`, `N`.

It prints each row like:

```text
Achilles   | Iliad   | paragraph     | mean=0.618, std=0.119, N=168
Achilles   | Odyssey | paragraph     | mean=0.599, std=0.101, N=12
...
```

This lets you see, at a glance:

- Rough continuity levels per character and epic.

### 5.2. Key findings (in this notebook’s current state)

From the paragraph-level and segment-level analyses, plus the randomization tests, we can say:

- **Achilles vs Hector (Iliad)**:
  - Their average drift is **very similar**.
  - Randomization test shows no meaningful difference.
  - So we **cannot** honestly claim that Achilles is more semantically “volatile” than Hector from these metrics alone.

- **Odysseus (Iliad vs Odyssey)**:
  - In both the main model and the alternate model, Odysseus in the *Odyssey* shows **lower average local similarity** than in the *Iliad*.
  - p-values are very small, suggesting this is unlikely to be random.
  - This is consistent with the view that Odysseus’s own epic is more **episodically varied**: he moves through more distinct narrative contexts.

- **Athena and Zeus**:
  - Show somewhat lower continuity than focused mortals (and appear in more episodes), which matches the idea that gods move across human storylines.
  - However, entity detection and episode labeling are noisy; we treat this as **suggestive**, not definitive.

- **Drift metrics and discourse mode**:
  - Embedding and lexical similarities correlate moderately for major characters.
  - Many large shifts happen at **speech↔narration transitions**, hinting at narratological structure affecting drift.

---

## 6. Challenges, Design Choices, and “Why This Way?”

### 6.1. Challenge: character detection

- **Problem:** Characters appear by name, epithet, pronoun, and sometimes ambiguous phrases (“son of Atreus”).
- **Naive approach:** Pure substring search on names only — this misses many occurrences.
- **Chosen compromise:**
  - Use regex patterns for:
    - Name.
    - Common epithets.
    - Translation-specific names (e.g. Jove for Zeus).
  - Add a minimal pronoun heuristic to catch unambiguous follow-ups.
- **Why not full-blown coreference / NER?**
  - Off-the-shelf English coref/NER is not tuned for Homeric translation style and may do poorly out-of-the-box.
  - Implementing a robust custom pipeline is a substantial research project itself.
  - For this notebook’s scope, a conservative, documented heuristic offers a good **cost–benefit trade-off**.

### 6.2. Challenge: episodes and structure

- **Problem:** We want to relate numeric drift changes to **Homeric episodes** (Cyclops, Nekyia, etc.), but:
  - The text is just paragraphs with some “Book X” headings.
- **Chosen approach:**
  - Use a hand-curated mapping from book numbers to short episode labels for a subset of books.
  - Attach these to paragraphs, and:
    - Annotate extreme drift points with them.
    - Compare within vs across episodes *where labels exist*.
- **Why not a full scene segmentation model?**
  - Scene segmentation in epic verse is non-trivial and not uniquely defined.
  - Anchoring on book-level episodes is a simpler and reasonably standard approximation.

### 6.3. Challenge: model dependence and robustness

- **Problem:** Any conclusion about “drift” could be an artifact of:
  - One particular embedding model.
  - One particular segmentation.
- **Chosen approach:**
  - Use a mainstream, well-tested sentence transformer as the base (`all-MiniLM-L6-v2`).
  - For at least one key result (Odysseus Iliad vs Odyssey), **recompute** with another model (`all-MiniLM-L12-v2`).
  - Check whether the **direction and significance** of the difference are stable.
- **Why not many models?**
  - Embedding computation is expensive.
  - For demonstration/teaching, one robustness check is enough to show that the main result is **unlikely to be pure model noise**.

### 6.4. Challenge: interpretability for non-experts

- Numerical results (means, stds, p-values) can be opaque to readers unfamiliar with stats or embeddings.
- The notebook addresses this by:
  - Printing **textual examples** of high/low-drift transitions.
  - Explaining in **natural language**:
    - What high similarity means: “two parts of the story feel similar”.
    - What low similarity means: “the story moves to a very different situation”.
  - Emphasizing **qualitative alignment** with known episodes.

---

## 7. How to Explain the Results to a Beginner

Imagine explaining this to someone who likes stories, not math:

- We track where each major character shows up in the two epics, **paragraph by paragraph**.
- For each place they appear, we ask a computer model to describe that paragraph as a **point in a big map of meanings**.
  - Paragraphs that talk about similar stuff end up near each other on this map.
- We then look at each character’s path through this map:
  - Does Achilles mostly move through places that look similar (fighting scenes, speeches), or does he jump around a lot?
  - Does Odysseus move through a **larger variety of places** than Achilles?
- When we compare:
  - Achilles and Hector turn out to have **very similar** patterns of change in the *Iliad*.
  - Odysseus in his own epic, the *Odyssey*, takes a **more varied route** through the map than in the *Iliad*.
- We also check that:
  - Big jumps in the map correspond to **big changes in the story**, like moving from one adventure to another.
  - These jumps often happen where scholars already say, “Here’s a new episode”.

So the notebook gives us a way to **visualize and measure**:

- How “smooth” or “choppy” each character’s story is.
- How that differs between the *Iliad* and the *Odyssey*.
- Whether our computer’s sense of “change” lines up with our human sense of episodes and scenes.

---

## 8. Final Thoughts

This notebook is best understood as:

- A **research lab** for exploring character-centered topic drift in Homer.
- A bridge between:
  - Classical philology (books, episodes, characters).
  - Modern NLP (embeddings, similarity, randomization tests).

It is **not** a final word on Homeric character design, but it:

- Demonstrates a careful, reproducible method.
- Shows that at least one intuitive claim (Odysseus’s greater episodic variety in the *Odyssey*) has **quantitative support** that is:
  - Stable across two embedding models.
  - Visible in both paragraph-level and segment-level analyses.
  - Grounded in actual text excerpts and known episodes.

For a data scientist, it is an example of:

- Building a pipeline end-to-end:
  - Data loading → preprocessing → modeling → visualization → statistics → interpretation.
- Tackling **real-world imperfections**:
  - Noisy entity detection.
  - Imperfect segmentation.
  - Model dependence.
- Being honest about **limitations** while still extracting **useful insights**.

For a beginner, it is a guided tour through:

- How computer models can “read” epic poetry.
- How to turn story structure into numbers.
- How to use those numbers to ask and tentatively answer interesting questions about characters and narratives.


