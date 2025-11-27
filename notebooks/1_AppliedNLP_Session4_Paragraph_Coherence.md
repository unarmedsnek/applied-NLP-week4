## 1\. What this notebook does

This notebook performs **paragraph\-level analysis** on two long texts:

- `../data/iliad.txt`
- `../data/odyssey.txt`

It computes:

- **Paragraph coherence** using MiniLM embeddings.
- **Topic clusters** using K\-Means.
- **3D semantic map** of paragraphs.
- **Similarity of book openings** across the 24 books of each epic.
- **Narrative progression** of style over the opening paragraphs.

It then visualizes results to compare the *Iliad* and the *Odyssey*.

---

## 2\. How to run this notebook

### 2\.1\. Requirements

Install dependencies in your environment (e.g., virtualenv, conda):

\```bash
pip install sentence-transformers numpy matplotlib scikit-learn scipy seaborn
\```

### 2\.2\. Expected files and folders

The notebook expects:

- Text data in `../data/`:
  \[...\]
- It writes output images to `../results/`.

Make sure these paths exist relative to the notebook.

### 2\.3\. How to run

1. Open the notebook `notebooks/1_AppliedNLP_Session4_Paragraph_Coherence.ipynb` in PyCharm or Jupyter.
2. Run cells **from top to bottom**.
3. Watch the console/log messages:
   \[...\]
4. Check `../results/` for the generated `.png` plots.

---

## 3\. Step\-by\-step code explanation

In this section, each block is explained in both **data scientist** terms and **beginner\-friendly** terms.

---

### 3\.1\. Imports and model setup

**What this block does**

- Imports standard Python libraries (regex, paths, plotting, math).
- Imports the `SentenceTransformer` model (MiniLM).
- Ensures a `../results` folder exists.
- Loads the MiniLM embedding model.

**Why these choices**

- `all-MiniLM-L6-v2` is:
  \[...\]

**Beginner explanation**

You are telling Python:

- `I will need tools for text, numbers, and graphs.`
- `I will also load a pre\-trained AI model that can convert text into vectors (lists of numbers) that represent meaning.`
- You create a folder to store the pictures produced later.

---

### 3\.2\. `load_book`: cleaning raw text files

**Function**

`def load_book(filepath: str) -> str:`

**What it does**

- Opens the raw Project Gutenberg text file.
- Removes the usual Gutenberg header and footer.
- Starts from the first meaningful marker (`BOOK I`, `CHAPTER I`, or `*** START OF`).
- Ends before the Gutenberg footer (`*** END OF`, `End of Project Gutenberg`).
- Returns cleaned text as a single string.

**Why this matters**

- Raw text files contain licensing text, metadata, and other non\-book content.
- If you analyze that noise, your statistics and clusters become meaningless.
- For any NLP project, **cleaning input** is a critical first step.

**Beginner explanation**

You are basically saying:
`Ignore all the extra rubbish before and after the actual book, and just keep the real story.`

**Challenges and choices**

- **Challenge**: Gutenberg headers vary between books.
- **Choice**: Use simple `if` checks for common markers instead of a complicated generalized parser; this is enough for this specific project.
- **Alternative**: Use a dedicated Project Gutenberg parser library. That would be more robust but adds dependency and complexity.

---

### 3\.3\. Load Homer's epics

`iliad_text = load_book('../data/iliad.txt')`  
`odyssey_text = load_book('../data/odyssey.txt')`

**What it does**

- Calls `load_book` twice and prints character counts.

**Data scientist view**

- At this point, you have two large strings with mostly clean narrative content.
- Counts help you sanity\-check: if the size looks way too small, something is wrong with cleaning.

**Beginner view**

You are now reading the two books from disk into memory and confirming what you loaded is big enough to be the full book.

---

### 3\.4\. `split_into_paragraphs`

**Function**

`def split_into_paragraphs(text: str, min_words: int = 10) -> List[str]:`

**What it does**

1. Normalizes newlines to `\n`.
2. Splits the text on **blank lines** (one or more empty lines) to get raw paragraphs.
3. Cleans extra spaces inside each paragraph.
4. Filters out short paragraphs (`< min_words`), e.g., chapter titles, section headers.
5. Returns a list of paragraph strings.

**Why this design**

- Most Gutenberg texts separate paragraphs with blank lines.
- Very short lines often are not real paragraphs (e.g., `BOOK I`, `CHAPTER IV`) and would distort coherence scores.
- `min_words = 10` is a heuristic: enough words to be a meaningful paragraph.

**Data scientist view**

- You define observation units: a **paragraph** is your basic analytic unit.
- Filtering out small text segments reduces noise in coherence and clustering.

**Beginner view**

Imagine you take the whole book and cut it every time there is an empty line: that's a paragraph.  
You then throw away very short cuts that are probably just titles.

---

### 3\.5\. `sentence_split`

**Function**

`def sentence_split(paragraph: str) -> List[str]:`

**What it does**

- Splits a paragraph into sentences using a regex on `. `, `! `, `?`.
- Strips whitespace and removes empty strings.

**Why this simple approach**

- It is quick and has **no extra dependencies**.
- For a rough coherence measure, perfect sentence boundaries are not strictly required.

**Limitation (challenge)**

- It fails on abbreviations (`Mr.`, `Dr.`), decimals, and some complex punctuation. This can:
  \[...\]

**Alternative**

- Use `nltk.sent_tokenize` or spaCy:
  \[...\]

**Beginner view**

You are cutting each paragraph into sentences based on ending punctuation.  
It is not perfect, but good enough to see overall patterns.

---

### 3\.6\. `cosine_similarity`

**Function**

`def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:`

**What it does**

- Computes the **cosine similarity** between two vectors:
  \[...\]

**Data scientist view**

- Cosine similarity is the standard for comparing **embeddings**.
- It is scale\-invariant: the magnitude of the vector does not matter, only direction, which is what we want for semantic comparison.

**Beginner view**

Each sentence is turned into a list of numbers.  
`cosine_similarity` measures **how close the angle** between two such lists is.  
The closer to 1, the more similar the meanings.

---

### 3\.7\. `paragraph_coherence_embeddings`

**Function**

`def paragraph_coherence_embeddings(paragraphs: List[str]) -> Tuple[list, list]:`

**High\-level**

For each paragraph:

1. Split into sentences.
2. Get a semantic embedding for each sentence using the model.
3. Compute all pairwise cosine similarities between sentence embeddings.
4. Take the **average** of those similarities as the **coherence** of that paragraph.
5. Also record the paragraph length (in words).

Returns:

- `scores`: list of coherence scores, one per paragraph.
- `lengths`: list of paragraph lengths in words.

**Detailed logic**

- Prints progress every 100 paragraphs (helps debug long runs).
- Skips paragraphs with \< 2 sentences (no pairwise comparisons possible).
- Uses `normalize_embeddings=True` in `model.encode`:
  \[...\]
- Loops `j` from 0 to `n_sents - 1` and `k` from `j+1` to `n_sents - 1`:
  \[...\]
- Averages them to get paragraph coherence.

**Why this design**

- Pairwise sentence similarity is a natural definition of **intra\-paragraph coherence**:
  \[...\]

**Data scientist view**

- Coherence score (for paragraph `p`) \= mean\_{i\<j} cos(s\_i, s\_j).
- This is an embedding\-based alternative to older topical coherence metrics like entity grids or topic models.
- You also append `len(" ".join(sents).split())` to capture paragraph length for further analysis.

**Beginner view**

For each paragraph:

> Turn each sentence into a `meaning vector`.  
> Compare every pair of sentences: if they are similar, give a high score; if different, lower.  
> Average all these pair scores to say how `focused` this paragraph is.

**Challenges and performance**

- **Computational cost**: pairwise similarity is O(n²) sentences per paragraph.
- With many sentences, this can be slow, but paragraphs usually have a small number (e.g. 2\-10), so it is fine.
- **Alternative**:
  \[...\]

This implementation favours **clarity** and **explicitness** over micro\-optimizations, which is good for a teaching notebook.

---

### 3\.8\. Splitting texts into paragraphs and computing coherence

`iliad_paras = split_into_paragraphs(iliad_text)`  
`odyssey_paras = split_into_paragraphs(odyssey_text)`

`iliad_scores, iliad_lengths = paragraph_coherence_embeddings(iliad_paras)`  
`odyssey_scores, odyssey_lengths = paragraph_coherence_embeddings(odyssey_paras)`

**What this does**

- Turns each full epic into a list of paragraphs.
- Computes coherence and length for each paragraph, separately for *Iliad* and *Odyssey*.

**Interpretation**

- After this step you have:
  \[...\]

---

### 3\.9\. Basic coherence summary

The notebook prints:

- Mean coherence for each epic.
- Explanation: higher \= more single\-topic, lower \= more multi\-topic paragraphs.

**Typical results (your run)**

- `Iliad` mean \~ 0\.296  
- `Odyssey` mean \~ 0\.309

**Data scientist view**

- Coherence values \~0\.3 indicate that on average, sentences within a paragraph are **moderately similar**, but not extremely tight.
- This matches expectations for epic poetry: paragraphs often mix:
  \[...\]

**Beginner view**

A score around 0\.3 means:

> On average, the sentences in a paragraph are related, but they are not all saying exactly the same thing.  
> These paragraphs often wander a bit, which is normal for epic stories.

---

### 3\.10\. Graph 1: `paragraph_coherence_vs_length.png`

**Code behaviour**

- Creates 2 scatter plots (one per epic) of:
  \[...\]
  - y\-axis: coherence score (average similarity between sentences).
- Uses different colors for *Iliad* and *Odyssey*.
- Saves to `../results/paragraph_coherence_vs_length.png`.

**What this reveals (data scientist)**

- Does longer text lead to lower coherence (more room to drift)?
- Or do longer paragraphs stay focused (e.g., long speeches)?
- You may observe:
  - Coherence concentrated around 0\.2\-0\.4 across a wide length range.
  - Some short paragraphs with very high coherence (sharp, focused statements).
  - Some long but coherent paragraphs (long but consistent descriptions).

**What this means (beginner)**

Think of each dot as a paragraph:

- Further right \= longer paragraph.
- Higher up \= more focused on one topic.

This graph shows if **longer paragraphs tend to lose focus** or not.

For epic poetry:

> You usually see many paragraphs with **medium length and medium coherence**, which matches the digressive style.

---

### 3\.11\. Graph 2: `paragraph_coherence_distribution.png`

**Code behaviour**

- Merges all scores to determine global min/max.
- Plots 2 histograms side by side:
  - Left: *Iliad* coherence scores.
  - Right: *Odyssey* coherence scores.
- Adds vertical dashed lines for the mean coherence in each epic.
- Saves to `../results/paragraph_coherence_distribution.png`.

**What this reveals (data scientist)**

- Both distributions cluster in 0\.2\-0\.4.
- Means:
  - *Iliad*: \~0\.296
  - *Odyssey*: \~0\.309
- The shapes are similar \-\> indicates **similar narrative style** (multi\-topic paragraphs) in both epics.

**What this means (beginner)**

The histogram shows:

- How many paragraphs have low vs high coherence.
- Most bars are in the **0\.2\-0\.4** range, so:

> Most paragraphs have **some structure**, but they also mix several ideas, which fits long storytelling.

---

### 3\.12\. Topic clustering (Graph 3: `paragraph_topic_clustering.png`)

**Key function**

`def cluster_paragraphs_by_topic(paragraphs: List[str], n_clusters: int = 5):`

**Steps**

1. Get a single embedding per paragraph (by encoding the whole paragraph text).
2. Run `KMeans(n_clusters=5, random_state=42, n_init=10)` on paragraph embeddings.
3. Return cluster labels (0\-4) for each paragraph.
4. Count how many paragraphs fall into each cluster for each epic.
5. Print sample paragraphs from each cluster to inspect themes.
6. Create pie charts of cluster proportions using manually editable labels in:
   - `iliad_topic_labels`
   - `odyssey_topic_labels`
7. Save figure as `../results/paragraph_topic_clustering.png`.

**Why 5 clusters**

- Heuristic: 5 is a reasonable number for **major themes**:
  - battle, dialogue, divine intervention, descriptions, transitions, etc.
- For teaching, 5 is easy to interpret.

**Data scientist view**

- This is unsupervised **semantic clustering** of paragraphs.
- The embedding model encodes semantics; K\-Means finds centroid\-based clusters in that space.
- Cluster counts and sample paragraphs help you **interpret** each cluster:
  - Cluster 0: mostly battle descriptions
  - Cluster 1: dialogue
  - Cluster 2: gods and divine intervention
  - etc.

You then manually rename clusters via `iliad_topic_labels` and `odyssey_topic_labels`.

**Beginner view**

You let the AI group paragraphs that `feel similar` in meaning, without telling it any labels.  
Then you read a few examples from each group and give them names like:

- `Battle scenes`
- `Conversations`
- `Descriptions of places`
- `Interventions of the gods`

**Why it is relevant**

- Shows that **both** epics share similar thematic structures and relative proportions (e.g., \~25\% battle, \~20\% dialogue).
- For data science, this is a case of **unsupervised topic discovery** using embeddings.

---

### 3\.13\. 3D semantic space (Graph 4: `paragraph_3d_analysis.png`)

**Process**

1. Encode up to 500 paragraphs each from *Iliad* and *Odyssey* for speed.
2. Stack them into `all_embeddings`.
3. Apply PCA with `n_components=3` to project high\-dimensional vectors into 3D.
4. Split the transformed embeddings back into `iliad_3d` and `odyssey_3d`.
5. Plot them using a 3D scatter plot:
   - Red circles \= *Iliad* paragraphs
   - Teal triangles \= *Odyssey* paragraphs
6. Save figure as `../results/paragraph_3d_analysis.png`.
7. Print explained variance (how much information the 3 components retain).

**Why PCA and 3D**

- Embeddings are typically 384 dimensions (for `all-MiniLM-L6-v2`).
- Humans cannot visualize 384D, so PCA finds three **principal components** that explain most variance.
- This is a standard technique to **visualize high\-dimensional data**.

**What it shows (data scientist)**

- Overlapping clouds indicate shared semantic themes across epics (warfare, gods).
- Distinct clusters or separated regions indicate unique content, e.g., *Odyssey*'s journey episodes vs *Iliad*'s focused Trojan War scenes.
- Density and shape differences show diversity vs focus.

**Beginner view**

Imagine each paragraph is a dot in 3D space.  
Paragraphs about similar things end up near each other.

- If red and teal are mixed in an area, both books talk about similar topics there.
- If red dominates one side and teal another, those are themes more unique to one book.

---

### 3\.14\. Heatmaps of book openings (Graph 5: `book_opening_similarity_heatmaps.png`)

**Function**

`def get_first_paragraph_per_book(text: str, epic_name: str) -> List[str]:`

**What it does**

1. Uses a regex `book_pattern = r'BOOK\\s+([IVXLCDM]+)'` to find all book headings.
2. For each book:
   - Takes its chunk of text.
   - Runs `split_into_paragraphs` to get paragraphs.
   - Picks the **first valid paragraph** as that book's opening.
   - If no paragraphs are found, falls back to the first 500 characters.
3. Returns a list of up to 24 first paragraphs (books I to XXIV).

**Then:**

- Encodes all first paragraphs for each epic using `model.encode`.
- Builds a similarity matrix (24×24) where entry (i,j) is:
  - `1 - cosine(embedding[i], embedding[j])` (cosine similarity)
- Plots two heatmaps using `seaborn.heatmap`:
  - Left: *Iliad* (Reds colormap).
  - Right: *Odyssey* (Blues colormap).
  - `vmin=0.3, vmax=1.0` to focus on moderate to high similarity.
  - Book labels I, II, ..., XXIV on both axes.
- Saves as `../results/book_opening_similarity_heatmaps.png`.

**Color interpretation**

- **Darker** colors \= **higher semantic similarity** (more similar).
- **Lighter** colors \= less similar.

**What this reveals (data scientist)**

- For *Iliad*:
  - Darker patches (blocks) indicate sets of books whose first paragraphs are semantically similar.
  - This suggests strong thematic grouping or repeated formulaic openings, especially around battle sequences.

- For *Odyssey*:
  - More variation, more lighter regions, meaning book openings are more varied.
  - This matches the **episodic** journey structure of *Odyssey*: each book often opens in a different context or location.

**Beginner view**

Think of each book as a chapter with an opening paragraph.  
The heatmap compares every opening paragraph with every other.

- Dark squares \= two books that start in a very similar way.
- Light squares \= two books that start very differently.

So:

> *Iliad* has more dark clusters \-\> some books start very similarly (e.g., battle intros).  
> *Odyssey* has more variation \-\> each adventure starts in its own style.

---

### 3\.15\. Narrative progression (Graph 6: `narrative_progression_analysis.png`)

**Function**

`def analyze_narrative_progression(paragraphs: List[str], epic_name: str, n_paragraphs: int = 100):`

**Steps**

1. Takes the first 100 paragraphs (`sample_paras`).
2. Encodes all 100 paragraphs into embeddings.
3. Computes a single **opening style vector** as the average of the first 5 paragraph embeddings.
4. For each of the 100 paragraphs:
   - Computes similarity to the opening style using `cosine_similarity`.
   - Stores these similarities.
5. Divides the 100 paragraphs into 4 sections:
   - Early (1\-25)
   - Early\-Mid (26\-50)
   - Late\-Mid (51\-75)
   - Late (76\-100)
6. Computes the average similarity for each section.

Returns:

- `all_similarities`: list of 100 similarity values.
- `section_averages`: dictionary with averages per section.
- `positions`: 1\-100 indices.

**Plotting**

- For each epic:
  - Plot similarity vs paragraph index (1\-100) as a line.
  - Add a smoothed trend line using `uniform_filter1d` (moving average).
  - Draw horizontal dashed lines for each section average.
  - Label section averages with their value.
- Save as `../results/narrative_progression_analysis.png`.

**What this shows (data scientist)**

- You are tracking **stylistic drift** over the first 100 paragraphs.
- If the trend stays flat and section averages are very similar (drift \< 0\.05), the author keeps a **consistent style**.
- If similarities drop significantly over sections, style diverges from the opening.

Your results show:

- Both *Iliad* and *Odyssey* have small drift (\< 0\.05).
- Indicates **stable narrative voice** across the early part of each epic.
- Minor dips correspond to shifts between:
  - Battle description vs dialogue vs descriptive passages.

**Beginner view**

You ask:

> `As we move from paragraph 1 to 100, does Homer write in the same way or change style a lot?`

You compare each later paragraph to the average style of the **first five** paragraphs.

- High similarity \= still close to the opening style.
- Low similarity \= more different.

The line mostly stays around the same level:

> So Homer keeps a fairly consistent way of writing, even while the story moves through different events.

---

### 3\.16\. Memory cleanup section

At the end there is an optional commented\-out block:

- Deletes large variables (`iliad_text`, `odyssey_text`, embeddings, scores).
- Closes plots.
- Deletes the `model`.
- Calls `gc.collect()` to free memory.

**Why it exists**

- On resource\-constrained machines, storing full texts, all embeddings, and all plots can consume a lot of RAM.
- This provides a manual `clean up memory` step.

**Beginner view**

If your computer gets slow or runs out of memory, you can uncomment this block and run it to free up space.

---

## 4\. Interpreting the results \- both as a data scientist and in simple terms

### 4\.1\. Summary of key quantitative findings

- **Coherence scores**:
  - *Iliad* mean \~ 0\.296
  - *Odyssey* mean \~ 0\.309
  - Most paragraphs cluster in 0\.2\-0\.4 range.

- **Topic clustering**:
  - 5 main semantic clusters in each epic.
  - Similar proportions (e.g., \~25\% battle/action, \~20\% dialogue, \~15\% divine, \~20\% description, \~20\% transitions).

- **3D embedding space**:
  - Overlapping regions: shared themes (warfare, gods, heroic exploits).
  - Distinct regions: unique content (e.g., *Odyssey*'s travel episodes).

- **Book opening similarity heatmaps**:
  - *Iliad*: more dark patches (high similarity), implying recurring opening patterns and thematic grouping.
  - *Odyssey*: more varied patterns, matching its episodic nature.

- **Narrative progression**:
  - Style drift \< 0\.05 across sections for both epics.
  - Suggests a consistent narrative voice.

---

### 4\.2\. Why these results are relevant for data science

1. **Chunk quality for RAG / LLMs**

   - Paragraphs with higher coherence make **better chunks** for Retrieval\-Augmented Generation (RAG):
     \[...\]
   - The notebook shows how to:
     \[...\]

2. **Representation learning in practice**

   - Demonstrates how **pre\-trained embedding models** can capture semantics without any task\-specific training.
   - You can reuse the same pipeline on any other documents, not just Homer.

3. **Unsupervised structure discovery**

   - K\-Means clustering on embeddings discovers **latent topics**.
   - PCA \+ 3D plots visualize **global semantic structure** of a corpus.
   - Heatmaps and progression plots show **sequence behavior** and **stylistic stability**.

4. **Scalability considerations**

   - Shows where computations are O(n²) and how that might impact performance.
   - Highlights the importance of **caching** embeddings and using vectorized operations in real production code.

---

### 4\.3\. Why these results make sense in human/literary terms

- **Low to medium coherence** (\~0\.3) fits **epic poetry**:
  - Frequent digressions.
  - Mixing of dialogue, narrative, and descriptions.
  - Invocation of gods and references to prior events.

- **Similar coherence distributions** across *Iliad* and *Odyssey*:
  - Same authorial tradition and style.
  - Comparable narrative structure with many mixed\-topic paragraphs.

- **Topic cluster similarities**:
  - Both epics share core elements:
    \[...\]
  - This matches scholarly understanding of Homeric poetry.

- **Heatmaps: *Iliad* more clustered, *Odyssey* more varied**:
  - *Iliad*: focus on Trojan War battles \-\> similar openings and repeated formulae.
  - *Odyssey*: Odysseus' journey from place to place \-\> openings vary more depending on where he is and who appears.

- **Stable narrative style**:
  - Oral\-formulaic tradition typically uses repeated phrases and structures.
  - The low drift captured numerically matches this tradition nicely.

---

### 4\.4\. Challenges and design choices recap

1. **Sentence splitting quality vs simplicity**

   - Chose a simple regex for clarity and zero external dependencies.
   - Trade\-off: slightly noisy sentence boundaries.
   - For production or research, a more robust tokenizer is recommended.

2. **Pairwise coherence computation cost**

   - O(n²) sentence pairs per paragraph; acceptable because paragraphs are small.
   - A vectorized approach or comparing each sentence to the mean embedding would be faster but less explicit for teaching.

3. **Number of clusters (K\-Means)**

   - Chosen as 5 for interpretability, not statistically optimized.
   - Could be tuned using silhouette scores, elbow method, etc.

4. **PCA to 3 components**

   - PCA is linear; might not perfectly capture nonlinear structure.
   - Chosen for simplicity and interpretability.

5. **Heatmap color interpretation**

   - Important to clearly state **darker \= more similar**.
   - Misinterpreting color intensity would invert conclusions.

6. **Narrative progression limited to first 100 paragraphs**

   - Chosen for runtime and clarity.
   - For deeper analysis, you could slide a window across the entire book.

---

## 5\. How to reuse/extend this notebook for other projects

Even as a beginner, you can reuse this pattern:

1. Replace `../data/iliad.txt` and `../data/odyssey.txt` with your own `.txt` files.
2. Keep the same functions:
   - `load_book` (you may tweak markers).
   - `split_into_paragraphs` and `sentence_split`.
   - `paragraph_coherence_embeddings`.
3. Re\-run the notebook to:
   - Analyze coherence distributions.
   - Visualize topic clusters.
   - Plot a 3D semantic map of your documents.
   - Investigate chapter\-opening similarities.
   - Explore narrative progression of style.

This gives you a **general recipe** for semantic analysis of long documents using embeddings, understandable even to newcomers but rigorous enough for data scientists.

---

**In summary:**  
The notebook is a full pipeline from raw text to rich, interpretable visualizations of semantic structure and coherence, demonstrating modern embedding\-based NLP techniques on a classic literary corpus, and this `README\-style` section walks through every step and design choice in detail for both beginners and advanced users.