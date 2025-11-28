## Discourse Marker Density Across the Iliad and Odyssey — Notebook Guide

This document explains the notebook `3_AppliedNLP_Session4_Discourse_Markers.ipynb` in **painstaking detail**.  

It is written for:

- **Beginners** who want a step-by-step explanation of what the code does and why.  
- **Data scientists** who care about the modeling choices, limitations, and how to interpret the results.  
- **Humanities / literature readers** who want to understand what these numbers say about the *Iliad* and *Odyssey*.

The notebook studies **“Discourse Markers”** along in Homer’s *Iliad* and *Odyssey* using modern NLP tools.

---

## 1. Overview of the Notebook
### 1.1 What this notebook does. (In simple terms)

- Loads English translations of Homer’s **Iliad** and **Odyssey** (Butler translation).
- Splits each epic into **paragraphs**, grouped by **book** (Book I, II, …).
- It detects any Discourse Markers that are already defined in a preset list. ("however", "although", etc.).
- It then does the following for each paragraph:
  - Finds the number of words.
  - Finds the number of markers.
  - Finds the marker-per-word density.
- It compares the distribution between the two epics.
- It then plots the following:
  - Scatter Diagram of paragraph length vs. density.
  - Histogram of density distribution.

### 1.2 What the graphs mean.

- The scatter Plot between The Iliad and Odyssey shows the relationship between the number of markers and the number of words in each paragraph based on this formula:
$$
\text{density} = \frac{\text{number of markers}}{\text{number of words}}
$$

- Each point represents one paragraph from the Iliad or the Odyssey.
- x-axis: total number of words.
- y-axis: discourse marker density.
- The reason why the points form several curved “paths” is because you can see distinct diagonal bands on each plot.
These correspond to paragraphs containing:

  - 0 markers → flat line at 0
  - 1 marker → curve defined by 1/𝐿
  - 2 markers → curve defined by 2/𝐿
  - 3 markers → curve defined by 3/𝐿
  - 4 markers → curve defined by 4/𝐿

**What does 𝐿 mean?**
- 𝐿 = paragraph length (number of words)
- M = number of discourse markers

computing this:

$$
\text{density} = \frac{M}{L}
$$

## 2. Environment & How to Run
This is how to run the Notebook and it also outlines what the requirments are.

- Python 3.10+
- Libraries: re, typing, matplotlib.pyplot
- The Iliad and The Odyssey books must be available and in the same folder as the notebook.

## 3. Step-by-Step Walkthrough 

This section explains every part of the notebook.
It is written for:

- Beginners: who want to understand what each part of the code does.

- Data scientists: who care about the design choices and limitations.

- Literature / classics readers: who want to understand what these graphs say about Homeric narrative structure.

### 3.1. Loading the Books
The analysis begins by loading two English translations:

- Homer’s Iliad

- Homer’s Odyssey

Both are stored as .txt files.

The function:

```
def load_book(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    if 'CHAPTER I' in text:
        start = text.find('CHAPTER I')
        text = text[start:]
    elif '*** START OF' in text:
        start = text.find('*** START OF')
        text = text[start + 100:]

    if '*** END OF' in text:
        end = text.find('*** END OF')
        text = text[:end]
    elif 'End of Project Gutenberg' in text:
        end = text.find('End of Project Gutenberg')
        text = text[:end]

    return text.strip()
```
\
**Why is this needed?**
- Project Gutenberg (where we got the texts from) texts contain large blocks of metadata at the start and end like:
  - License information
  - Scanner credits
  - Editor notes

- The function trims these using predictable markers:

```
"CHAPTER I"

"*** START OF"

"*** END OF"
```

- This ensures that only the actual content of the epic is analyzed.

**What is the outcome?**
- We get two cleaned strings:
  - Iliad_text
  - Odyssey_text
- Which contwain only the story itself.

### 3.2. Splitting into Paragraphs

```
def split_into_paragraphs(text: str, min_words: int = 10) -> List[str]:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    raw_paras = re.split(r'\n\s*\n+', text)
    ...
```

**What does this do?**

- Normalize newlines
  - Gutenberg files sometimes mix \r, \r\n, and \n.
  - Standardizing prevents accidental paragraph merges.

- Split into paragraphs.
  - We use blank lines (\n\n) to mark paragraph boundaries:

```
re.split(r'\n\s*\n+', text)
```

- Clean whitespace
  - Removing extra spaces and line breaks inside paragraphs.

- Filter out short paragraphs
  - Any paragraph with under 10 words is discarded:
  
```
if len(cleaned.split()) < min_words:
    continue
```

**Why paragraphs?**

- Paragraphs are a reasonable unit for analyzing structure in prose translations.
- They correspond to shifts in:
  - Scene
  - Speaker
  - Narrative focus

This is important when studying discourse markers, which tend to be used at topic boundaries.

### 3.3. Defining Discourse Markers

we define a list of markers that we want to detect:

```
DISCOURSE_MARKERS = [
    'however', 'therefore', 'moreover', 'meanwhile', 'suddenly',
    'although', 'though', 'even though', 'in contrast', 'on the other hand',
    'for example', 'for instance', 'at the same time', 'finally',
    'in conclusion', 'nevertheless', 'nonetheless', 'instead', 'after all'
]
```

**Why these markers?**
- They represent explicit English connectives used to signal:

  - Contrast (however, on the other hand)

  - Causation / consequence (therefore)

  - Temporal transitions (meanwhile, suddenly)

  - Examples (for example)

These are common in modern narrative prose and academic writing, but much rarer in older or literal translations of ancient texts.

### 3.4. Counting Markers

This function counts markers in a paragraph:

```
def count_markers(paragraph: str) -> int:
    text = paragraph.lower()
    count = 0
    for m in DISCOURSE_MARKERS:
        count += text.count(m)
    return count
```

**What happens here?**
- Convert to lowercase
  - This makes matching case-insensitive.

- Count each marker using .count()
  - This is simple substring matching.

**Density formula:**
\
For each paragraph you do the following:

- density = $$
\text{density} = \frac{\text{number of discourse markers}}{\text{number of words}}
$$

- The code implements this inside:

```
markers = count_markers(p)
densities.append(markers / n_words)
```

**Expected outcome:**
\
Because discourse markers are extremely rare in Samuel Butler’s translation:

- Most paragraphs have 0 markers

- Some have 1

- Very few have 2–4

### 3.5. Computing Density for Each Epic

The workflow is parallel:

```
Iliad_paras = split_into_paragraphs(Iliad_text)
Odyssey_paras = split_into_paragraphs(Odyssey_text)

w_dens, w_len = marker_density(Iliad_paras)
g_dens, g_len = marker_density(Odyssey_paras)
```

The table gives us the outputs and what they mean:

| Output | Meaning |
|--------|---------|
| `w_dens` | Discourse marker density per paragraph (Iliad) |
| `w_len`  | Paragraph word counts (Iliad) |
| `g_dens` | Discourse marker density per paragraph (Odyssey) |
| `g_len`  | Paragraph word counts (Odyssey) |

## 4. Visualizations
### 4.1. Scatter Plots: Paragraph Length vs Density

The scatter plots show:
- x-axis: paragraph length
- y-axis: density

**Why the diagonal “paths” appear**

Because density is:

density = $$
\text{density} = \frac{M}{L}, \quad M = 0, 1, 2, 3\ldots
$$


The fixed values of M create hyperbolic curves:
- 0 markers → horizontal line at 0
- 1 marker → 1/L curve
- 2 markers → 2/L curve
- 3 markers → 3/L curve

These curves visibly appear as separate “bands” in your plot.

**Interpretation**
- Most paragraphs fall on the 0/L line.
- A smaller number follow the 1/L curve.
- Very few appear in the 2/L or 3/L bands.

### 4.2. Histograms: Distribution of Density

The histogram overlays both epics.
**Key patterns**
- Huge spike near zero
  - Most paragraphs contain 0 markers, so density ≈ 0.
- Long right tail
  - A small number of paragraphs contain 1–4 markers.
- Overlapping distributions
  - The Iliad and Odyssey histograms almost perfectly overlap.

**Interpretation**
- Both texts use explicit connective markers extremely rarely.
- This matches expectations for Butler’s translation style.

## 5. Interpretation of Results

**1. Both epics have very low discourse-marker usage**
\
Explicit markers like "however" or "therefore" are a modern device.
Their absence is normal for:
- Epic poetry
- 19th-century translations
- Narrative styles based on parataxis (“and then…”)

**2. Most paragraphs contain zero markers**
\
- The histogram spike at 0 confirms this.

**3. The curves in the scatter plot are exactly density = 1/L, 2/L, 3/L**
\
- This shows the mathematics is working exactly as intended.

**4. The two epics behave almost identically**
\
Because:
- Same translator
- Similar narrative conventions
- Similar paragraph structure

## 6. Limitations and Caveats


**1. The markers are English-specific**
\
The Greek texts contain a rich set of particles (δέ, γάρ, μὲν, ἀλλά).
None of these are represented in the English translation.

**2. Translation effects**
\
Samuel Butler’s translation uses a very direct, literal, Victorian prose style.
It avoids explicit connectors.

**3. Paragraph structure is editorial**
\
Gutenberg paragraphs do not necessarily match ancient Greek discourse units.

**4. Simple substring matching may miscount**
- .count() does not enforce word boundaries, so:

- “although” is fine

- but phrases like “however,” with punctuation depend on punctuation being retained

- multi-word markers may be missed if line breaks intervene

**5. Not all discourse signals are lexical**
\
Greek and English both use:

-Syntax
\
-Topic-shifting constructions
\
-Narrative pacing

## 7. Summary & Possible Extensions
**Summary**

- Both epics show extremely low discourse marker density.

- Most paragraphs contain 0 markers; a few contain 1–2.

- Scatter plots show clear 1/L, 2/L, 3/L curves.

- The distribution is almost identical between the Iliad and the Odyssey.

**Possible Extensions**

**1. Compare translations**
\
e.g., Butler vs Fagles vs Lattimore.

**2. Detect Greek discourse particles**
\
Analyze original Greek markers such as:

- δέ
- γάρ
- μέν
- ἀλλά

**3. Sentence-level analysis**
\
Instead of paragraphs, compute density per sentence.

**4. POS-tagging and dependency parsing**
\
Identify structural discourse cues beyond lexical markers.

**5. Machine-learning classification**
\
Predict episode boundaries using discourse markers + other features.