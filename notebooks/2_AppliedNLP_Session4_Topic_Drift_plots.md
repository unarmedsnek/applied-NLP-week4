## Plot Guide for `2_AppliedNLP_Session4_Character_Centered_Topic_Drift.ipynb`

This file explains how to read the **presentation plots** in the notebook and what each one means for **Homer’s Iliad and Odyssey**.

For every plot there are two parts:
- **How to read the graph**: what the axes, shapes, and statistics mean.
- **What it means for the books**: how to interpret the pattern in terms of the stories.

---

## Quick stats cheatsheet

- **Cosine similarity** (0 to 1):
  - **1.0**: two paragraphs or segments are **very similar** in meaning.
  - **0.0**: they are **completely different**.
  - In our context: *higher similarity = more local continuity; lower similarity = stronger topic drift*.

- **p-value** (from the randomization test):
  - Question: “If there were **no real difference** between these two groups, how often would a difference this large show up just by chance?”
  - **Large p** (e.g. 0.5, 0.7): The observed difference is common under “no real difference” → we **cannot** say the groups differ.
  - **Small p** (e.g. 0.05, 0.01, 0.001): The observed difference is rare under “no real difference” → evidence of a **real difference**.

- **Pearson r** (correlation coefficient, between −1 and 1):
  - **r > 0**: as x increases, y tends to increase (positive relationship).
  - **r < 0**: as x increases, y tends to decrease (negative relationship).
  - **|r| near 0**: weak or no linear relationship.
  - In our context: positive r between embedding similarity and lexical similarity means both metrics tend to agree.

---

## Odysseus: Mean Local Topic Continuity in Iliad vs Odyssey

This is the **two-panel bar chart** comparing Odysseus in the two epics, using the main embedding model (left) and a second model for robustness (right).

### How to read this graph

- **X-axis** (in each panel):
  - Two bars: **Iliad** and **Odyssey**.
- **Y-axis**:
  - **Mean cosine similarity** between **consecutive Odysseus paragraphs** (paragraph-level drift).
  - Higher bar = Odysseus tends to move in **smaller steps** (more continuity).
  - Lower bar = Odysseus tends to jump into **more different situations** (more drift).
- **Left panel**:
  - Uses the main model `all-MiniLM-L6-v2`.
- **Right panel**:
  - Uses the alternate model `all-MiniLM-L12-v2` (robustness check).
- Error bars:
  - Show **standard deviation** → how spread out the similarities are around the mean.

### What it means for the books

- In **both panels**, the **Odyssey bar is lower** than the Iliad bar:
  - Odysseus’s local context (from one appearance to the next) changes **more** in the *Odyssey* than in the *Iliad*.
- This difference is **statistically significant** in the randomization test:
  - The p-values for “Odyssey vs Iliad” are very small → unlikely to be random.
- Because this pattern holds under **two different embedding models**, it’s not just a modeling artifact.
- **Interpretation**:
  - In the *Iliad*, Odysseus is woven into a relatively **steady war narrative**; each time he appears, the scene is not radically different.
  - In the *Odyssey*, each new appearance is more likely to be in a **distinct type of scene** (another island, another monster, another speech), matching the classic idea of the *Odyssey* as a **series of discrete adventures**.

---

## Achilles vs Hector Local Topic Continuity (Iliad)

This is the **boxplot** comparing paragraph-level drift for Achilles and Hector in the Iliad.

### How to read this graph

- **X-axis**:
  - Two boxes labeled **Achilles** and **Hector**.
- **Y-axis**:
  - Cosine similarity between **consecutive paragraphs** where that character appears (within the Iliad).
  - Higher values = more continuity, lower values = more topic drift.
- Each **box** shows the distribution:
  - The **middle line** = median similarity.
  - The box edges = middle 50% of values.
  - Whiskers = typical range; outliers are hidden for clarity.
- If the boxes are at similar heights and largely overlap, the two characters have **similar** local continuity.
- In the randomization test:
  - The p-value for Achilles vs Hector is **large** (~0.7), meaning the observed mean difference is what you’d expect from random fluctuation if there were no real difference.

### What it means for the books

- Achilles and Hector, in the *Iliad*, have **very similar** local topic continuity:
  - Both mostly move through similar battle/family scenes with roughly comparable semantic steps.
- This result **does not support** a claim like “Achilles is structurally far more volatile than Hector” at the scale we’re measuring.
- This is an important “negative control”:
  - It shows the method can say “**no real difference here**” for some comparisons, rather than forcing a story of contrast.
  - That builds trust in the more dramatic results (like Odysseus Iliad vs Odyssey).

---

## Within vs Across Episode Drift for Odysseus in the Odyssey

This is the **boxplot** with:
- “Within episode (N=…)”
- “Across episodes (N=…)”

### How to read this graph

- **X-axis**:
  - Left box: **Within episode** → pairs of Odysseus paragraphs where both belong to the **same labeled episode** (e.g. both in “Cyclops Polyphemus”).
  - Right box: **Across episodes** → pairs where the paragraphs belong to **different episodes** (e.g. one in “Cyclops”, one in “Circe”).
- **Y-axis**:
  - Cosine similarity between those **paragraph pairs**.
  - Higher = more continuity; lower = bigger jump in topic.
- **Within episode box**:
  - Uses many pairs (N shown under label).
  - Its median is typically **higher**.
- **Across episodes box**:
  - Often has far fewer pairs (N small), but its median and box are usually **lower**.
- The exact numbers will vary, but the key pattern is:
  - Within-episode similarities cluster higher than across-episode similarities.

### What it means for the books

- Inside a single named Odyssey episode:
  - Odysseus tends to move in **small steps**; his local context stays relatively coherent.
  - Example: within the Cyclops episode, consecutive Odysseus paragraphs feel related.
- When crossing from one episode to another:
  - Similarities **drop**; his context changes much more.
  - These are semantic “cuts” between big adventures (e.g. from Cyclops to Circe).
- This result shows that the drift metric respects **episode boundaries** that readers and scholars already recognize:
  - It is not just random noise; it tracks real narrative structure in the *Odyssey*.

---

## Context-Aware Topic Drift for Achilles in the Iliad

This is the **line plot** of Achilles’ paragraph-level drift in the Iliad, with **red dots** labeled by book/episode.

### How to read this graph

- **X-axis**:  
  “Consecutive character paragraph pair” for **Achilles** only.
  - 0 = first pair of consecutive Achilles paragraphs.
  - 1 = second pair, and so on.
- **Y-axis**:  
  Cosine similarity between those two paragraphs.
  - Around **0.6–0.8** → steady, moderate continuity.
  - Deep dips near **0.2–0.3** → very different context from one appearance to the next.
- **Blue line**:
  - Tracks how similar each step is over the course of Achilles’ appearances.
  - Shows many fluctuations, but mostly in a mid–high band.
- **Red points with labels**:
  - Mark the **lowest similarity points** (largest jumps).
  - The labels (e.g. “Book XIX”) show in which **book** those big shifts occur.

### What it means for the books

- Achilles’ storyline in the *Iliad* is mostly:
  - **Locally continuous**: his appearances tend to be in related battle scenes, speeches, or emotional moments.
  - The blue line rarely drops to extremely low values.
- The **red dips** pick out:
  - Major narrative transitions in his arc — points where his situation, tone, or role changes sharply (e.g. key battles, decisions to withdraw or re-enter combat, grief scenes).
- This plot acts like a **semantic heartbeat** of Achilles’ narrative:
  - Where the line is stable, his story is moving steadily.
  - Where the line plunges, something genuinely new is happening in his story.

---

## Segment-level Drift Correlation for Odysseus in the Odyssey

This is the **scatter plot** where each point is a pair of consecutive **Odysseus segments** in the Odyssey.

### How to read this graph

- **X-axis**:
  - `Embedding similarity (cosine)` between two consecutive Odysseus **segments**.
  - Segments are longer runs of paragraphs where Odysseus appears and no other tracked character does.
- **Y-axis**:
  - `Lexical similarity (Jaccard)` between the same segments.
  - Jaccard = overlap of important words: \(|A∩B| / |A∪B|\).
- Each **point**:
  - Represents one segment-to-segment transition.
- **Pearson r, p-value**:
  - r ~ positive value (e.g. 0.4–0.6): as embedding similarity increases, lexical similarity tends to increase.
  - p small: the correlation is very unlikely to be due to chance.

### What it means for the books

- When the model says two Odysseus segments are “similar” in embedding space:
  - They also tend to share many of the same **content words**.
- Conversely, when embeddings say “these segments are different,” their word sets tend to differ too.
- This is a **method validation** plot:
  - It shows that the drift is grounded in **real textual differences**, not arbitrary behavior of the embedding model.
  - That gives us more confidence that peaks and dips in drift truly reflect changes in the story.

---

## Odysseus: Share of Large Topic Jumps (Iliad vs Odyssey)

This is the **bar chart** comparing how often Odysseus experiences **big jumps** in topic in each epic.

### How to read this graph

- **X-axis**:
  - Two bars: **Iliad** and **Odyssey**.
- **Y-axis**:
  - `Fraction of pairs with sim < 0.5` (or another chosen threshold).
  - This is the **proportion** of Odysseus paragraph pairs whose similarity is below the threshold → treated as “large topic jumps”.
- Higher bar = Odysseus has **more big jumps** in that epic.

### What it means for the books

- The **Odyssey bar is higher**:
  - A larger share of Odysseus’s local steps in the *Odyssey* are big semantic jumps.
- The **Iliad bar is lower**:
  - In the *Iliad*, he is more often in sequences where his context changes in **smaller increments**.
- This neatly captures the intuitive distinction:
  - In the *Iliad*, Odysseus is part of a more **continuous war narrative**.
  - In the *Odyssey*, he repeatedly leaps from one distinct adventure or island to another — more **scene-to-scene jumps**.

---

## Odysseus in the Odyssey: Within vs Across Episode Continuity

This is the **bar chart** showing mean similarity **within** and **across** episodes for Odysseus in the Odyssey.

### How to read this graph

- **X-axis**:
  - Two bars: `Within episode` and `Across episodes`.
- **Y-axis**:
  - Mean cosine similarity between paragraph pairs.
- Each bar has an **error bar** (± standard deviation).
- Higher `Within episode` bar:
  - Local steps inside the same episode are more similar.
- Lower `Across episodes` bar:
  - Steps crossing from one episode label to another are less similar.

### What it means for the books

- Inside episodes like “Cyclops Polyphemus” or “Nekyia (Underworld)”:
  - Odysseus’s context evolves **smoothly**.
- When we move from one named episode to another:
  - The context changes more sharply.
- This confirms, in numbers, the traditional view of the *Odyssey* as a **sequence of well-formed episodes**:
  - High continuity within each adventure.
  - Larger shifts between adventures.

---

## Odysseus in the Odyssey: Local Continuity by Episode

This is the **line plot** where each point is a named Odyssey episode.

### How to read this graph

- **X-axis**:
  - Episode labels (e.g. “Telemachus at Ithaca”, “Cyclops Polyphemus”, “Nekyia (Underworld)”, …).
- **Y-axis**:
  - Mean cosine similarity for **Odysseus paragraph pairs whose midpoint falls in that episode**.
- Interpretation:
  - A higher point = **more local continuity** in that episode.
  - A slightly lower point = **more internal variety** in that episode.
- The overall line tends to stay around the mid-range (e.g. 0.5–0.6) and does not wildly swing.

### What it means for the books

- Across the named adventures, Odysseus’s story is **consistently coherent**:
  - No episode is wildly more chaotic at the local level.
  - Each adventure maintains a reasonably stable thematic and narrative focus.
- Small differences can be interesting:
  - Some episodes (e.g. Cyclops) may show slightly higher continuity.
  - Others (e.g. Nekyia) may have slightly lower continuity, reflecting internal shifts (multiple encounters, varied tones).
- Overall, this supports the idea that each episode is a **self-contained narrative block**, with its own internal cohesion.

---

## Odysseus: Book-by-Book Local Topic Continuity

This is the **two-line plot** with books on the X-axis, showing Odysseus’s local continuity per book in each epic.

### How to read this graph

- **X-axis**:
  - Book numbers (Roman numerals) where Odysseus has enough data.
  - Blue line: **Iliad** books.
  - Orange line: **Odyssey** books.
- **Y-axis**:
  - Mean cosine similarity for Odysseus paragraph pairs whose midpoint falls in that book.
- Each point:
  - Summarizes how “smooth” or “choppy” Odysseus’s local story is **within that specific book**.
- The Iliad line may have **fewer points** than the Odyssey line:
  - Because Odysseus appears in enough paragraphs to form pairs in fewer Iliad books.

### What it means for the books

- The **Iliad line is generally higher**:
  - When Odysseus appears in the *Iliad*, his local context inside a book tends to change less from step to step.
  - This matches his role as a relatively stable figure in a single, large-scale conflict.
- The **Odyssey line is lower overall and spans more books**:
  - Across many books of the *Odyssey*, Odysseus’s local steps are less similar.
  - This reflects the **restless movement** of his journey through multiple locations, challenges, and tones.
- Taken together with the other plots, this gives a “zoomed-out” view:
  - Book by book, the *Iliad* keeps Odysseus in a more unified narrative environment, while the *Odyssey* distributes him across many books with more varied, less tightly coupled scenes.

---

These plots, read together, provide a multi-scale picture of **character-centered topic drift**:
- From individual paragraph steps, to segments, to episodes, to whole books.
They show that Odysseus’s journey in the *Odyssey* really does have a more **episodic and varied structure**, while Achilles and Hector in the *Iliad* share a more **uniform local continuity**—all grounded in quantitative measures that respect the text’s words and episode structure.


