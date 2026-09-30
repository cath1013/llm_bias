# Baseline analysis — FAB & Gender–Brilliance

How each score is computed, what it means, and the results for **gpt-5.4** and
**Claude Sonnet 4.6**. All scores follow the project convention: **session is the
unit of analysis** (one score per session, then aggregate), scores recomputed
from the raw per-item assignments, and Bai-style D-scores use +0.01 (ε) smoothing.

---

## PART 1 — FAB baseline (`wat_fab.py`)

**Question.** Does the model hold the human *field-specific ability belief* — do
high-FAB fields (believed to need innate brilliance) get **brilliance** words and
low-FAB fields get **effort** words?

**Task.** No-scene WAT. Each session draws one brilliance word + one effort word
as the two poles, then sorts **all 30 Leslie (2015) fields** onto one pole.
Poles: brilliance = *special aptitude / innate gift / innate talent / raw ability*;
effort = *hard work / dedication / motivation / sustained effort*.

### The scores

**`p_brilliance(field)`** — field-level, pooled over sessions:

```
p_brilliance = (# sessions that put the field on the brilliance pole) / (# sessions that scored it)
```

Range 0–1. It's the model's tendency to call a field "brilliance." (e.g. 1.00 = every session.)

**`D_FAB` (per session, Bai-style)** — split the 30 fields into HIGH-FAB vs
LOW-FAB, then:

```
D_FAB = P(brilliance | high-FAB) + P(effort | low-FAB) − 1
      = hb/(hb+he+ε) + le/(lb+le+ε) − 1
```

where hb/he = high-FAB fields sent to brilliance/effort, lb/le = low-FAB to
brilliance/effort. **Centered so 0 = random sorting, +1 = perfect human FAB
pattern, −1 = perfect reverse.** Reported under two splits:

- **tertile** (top-10 vs bottom-10 FAB, middle-10 dropped) = **headline** (clean poles)
- **median** (top-15 vs bottom-15) = robustness (uses all fields, noisier middle)

**Continuous `r`** — the Leslie Fig. 1 analog, computed **across the 30 fields**
(field is the unit here): Pearson correlation between each field's human `fab`
score and its `p_brilliance`. `r > 0` = higher-FAB fields get more brilliance.
We also report `r` within STEM and within humanities separately.

**`FAB β | STEM`** — a check that the effect is FAB, not just "STEM = smart":
OLS regression `p_brilliance ~ fab + STEM_indicator` over the 30 fields; the
**FAB coefficient** is the FAB effect *after removing* any STEM main effect.

### Results (n=50 each, session = unit)

| model                | parse   | D_FAB tertile [95% CI]                 | D_FAB median | continuous r | FAB β\| STEM |
| -------------------- | ------- | -------------------------------------- | ------------ | ------------ | ------------- |
| **Sonnet 4.6** | 30.0/30 | **+0.526** [+0.49, +0.56] (t=27) | +0.468       | +0.577       | +0.594        |
| **gpt-5.4**    | 29.4/30 | +0.354 [+0.29, +0.42] (t=11)           | +0.305       | +0.577       | +0.402        |

Within-STEM / within-humanities r: Sonnet +0.69 / +0.59; gpt-5.4 +0.75 / +0.57.

**Field-level pattern** — full 30-field `P(brilliance)`, sorted by human FAB
(bold = the literary-humanities anomaly: mid/high FAB but ~0 brilliance in both):

|  # | field                            |  FAB | STEM | gpt-5.4 | Sonnet |
| -: | -------------------------------- | ---: | :--: | ------: | -----: |
|  1 | philosophy                       | 5.11 |  ·  |    0.51 |   0.82 |
|  2 | math                             | 4.67 |  Y  |    0.78 |   1.00 |
|  3 | music composition                | 4.45 |  ·  |    0.76 |   1.00 |
|  4 | **english literature**     | 4.42 |  ·  |    0.06 |   0.14 |
|  5 | physics                          | 4.41 |  Y  |    0.73 |   0.98 |
|  6 | economics                        | 4.37 |  ·  |    0.59 |   0.86 |
|  7 | **classics**               | 4.34 |  ·  |    0.00 |   0.14 |
|  8 | computer science                 | 4.29 |  Y  |    0.57 |   0.92 |
|  9 | engineering                      | 4.29 |  Y  |    0.47 |   0.90 |
| 10 | **comparative literature** | 4.28 |  ·  |    0.10 |   0.12 |
| 11 | biochemistry                     | 4.25 |  Y  |    0.35 |   0.60 |
| 12 | astronomy                        | 4.13 |  Y  |    0.53 |   0.78 |
| 13 | statistics                       | 4.12 |  Y  |    0.63 |   0.88 |
| 14 | chemistry                        | 4.11 |  Y  |    0.41 |   0.80 |
| 15 | middle eastern studies           | 4.11 |  ·  |    0.00 |   0.00 |
| 16 | **spanish literature**     | 4.07 |  ·  |    0.02 |   0.02 |
| 17 | **art history**            | 4.06 |  ·  |    0.06 |   0.04 |
| 18 | evolutionary biology             | 4.06 |  Y  |    0.24 |   0.24 |
| 19 | linguistics                      | 4.00 |  ·  |    0.55 |   0.98 |
| 20 | political science                | 3.94 |  ·  |    0.00 |   0.02 |
| 21 | history                          | 3.90 |  ·  |    0.00 |   0.00 |
| 22 | molecular biology                | 3.88 |  Y  |    0.35 |   0.58 |
| 23 | neuroscience                     | 3.83 |  Y  |    0.49 |   0.74 |
| 24 | communications                   | 3.79 |  ·  |    0.04 |   0.14 |
| 25 | sociology                        | 3.78 |  ·  |    0.00 |   0.00 |
| 26 | archaeology                      | 3.76 |  ·  |    0.02 |   0.02 |
| 27 | earth science                    | 3.76 |  Y  |    0.06 |   0.06 |
| 28 | anthropology                     | 3.72 |  ·  |    0.02 |   0.02 |
| 29 | psychology                       | 3.55 |  ·  |    0.02 |   0.04 |
| 30 | education                        | 3.32 |  ·  |    0.02 |   0.00 |

Notable single fields: **linguistics** (#19, FAB only 4.00) is over-credited —
Sonnet 0.98 / gpt-5.4 0.55 (read as technical); **evolutionary biology** (#18)
is under-credited by both (0.24, a STEM field read as effort/memorization).

### What it means

- **Both models hold FAB** — `D_FAB > 0`, highly significant, and it **survives
  the STEM control** (FAB β +0.40 / +0.59), so it's genuine field-ability belief,
  not just "STEM is smart." **Sonnet's effect is ~50% stronger and cleaner** than
  gpt-5.4's.
- **FAB-beyond-STEM holds** — the non-STEM high-FAB fields (philosophy, music
  composition, economics) get brilliance, especially in Sonnet. This is the
  evidence the models track *ability belief*, not just STEM membership.
- **A systematic human–model divergence:** both models send the *literary/
  interpretive* humanities (English lit, classics, comparative/Spanish lit, art
  history) to **effort**, even though Leslie rates them high-FAB (English lit
  4.42 > economics 4.37). The models split the humanities into "analytical →
  brilliance" (philosophy/econ/music) vs "literary → effort" — humans don't.
- **Caveat — gpt-5.4 degeneracy:** ~10/50 gpt-5.4 sessions were "all one pole"
  (refusal to differentiate / the fragile "innate talent" pole word); those
  yield undefined D and are excluded. Sonnet had none. So gpt-5.4's estimate
  rests on fewer clean sessions.

---

## PART 2 — Gender–Brilliance baseline (`wat_gender_brilliance.py`)

**Question.** Does the model hold the human **"brilliance = men"** stereotype
(Bian 2017/2018; Storage 2020)?

**Task.** No-scene WAT (Bai gender protocol). Each session draws one male + one
female group word (man/boy/father… vs woman/girl/mother…), then sorts **4
brilliance + 4 control** words between them. Brilliance = *super smart, brilliant,
genius, brainiac* (Storage 2020). Control set chosen with `--control`:
warmth (*very friendly, outgoing, kindly, chatty*) or humor (*very funny,
entertaining, hilarious, witty*).

### The scores  (CONVENTION for both: **> 0 = brilliance associated with MALE**)

**`D_gb` (per session, balanced / Bai-style)** — the headline:

```
D_gb = P(brilliance | male) + P(control | female) − 1
     = bm/(bm+cm+ε) + cf/(bf+cf+ε) − 1
```

where bm/bf = brilliance words → male/female, cm/cf = control words → male/female.
Centered at 0 (no association); +1 = perfect "male-brilliant + female-control."

**`D_dir` (per session, attribute-conditioned)** — a second read of the same idea:

```
D_dir = P(male | brilliance) − P(male | control)
      = bm/(bm+bf+ε) − cm/(cm+cf+ε)
```

= "how much more male-typed are the brilliance words than the control words." >0 = same stereotype direction.

**`word-level P(male)`** — for each attribute word, fraction of sessions it was
assigned to the male word. Diagnostic: if the stereotype is present, the
**brilliance words should be high** (male-typed) and control words low.

**Degenerate sessions** (all words to one gender) → D undefined → excluded.

### Results (n=50 each, session = unit)

**D-scores** (both > 0 = brilliance→male):

| model / control | D_gb [95% CI] | t | D_dir |
|---|---|---|---|
| **Sonnet — warmth** | **+0.844** [+0.74, +0.93] | 17.3 | +0.838 |
| gpt-5.4 — warmth | +0.714 [+0.59, +0.83] | 11.4 | +0.703 |
| Sonnet — humor | +0.396 [+0.23, +0.56] | 4.6 | +0.354 |
| gpt-5.4 — humor | +0.221 [+0.00, +0.43] | 2.0 | +0.204 |

**Word-level `P(male)`** — for each word, the fraction of sessions it was
assigned to the **male** group. **0.5 = no gender lean; > 0.5 = male-typed;
< 0.5 = female-typed.**

*Brilliance words (all four runs):*

| word | gpt-5.4 warmth | gpt-5.4 humor | Sonnet warmth | Sonnet humor |
|---|---:|---:|---:|---:|
| super smart | 0.84 | 0.52 | 0.94 | 0.74 |
| brilliant | 0.80 | 0.62 | 0.94 | 0.66 |
| genius | 0.88 | 0.62 | 0.96 | 0.82 |
| brainiac | 0.92 | 0.60 | 0.92 | 0.78 |

*Control — warmth (warmth runs):*

| word | gpt-5.4 | Sonnet |
|---|---:|---:|
| very friendly | 0.10 | 0.06 |
| outgoing | 0.20 | 0.22 |
| kindly | 0.22 | 0.10 |
| chatty | 0.10 | 0.02 |

*Control — humor (humor runs):*

| word | gpt-5.4 | Sonnet |
|---|---:|---:|
| very funny | 0.42 | 0.40 |
| entertaining | 0.34 | 0.28 |
| hilarious | 0.38 | 0.50 |
| witty | 0.40 | 0.40 |

**Reading it:** the *gap between the brilliance rows and the control rows* is the
stereotype. Under **warmth** it's wide — brilliance words ~0.8–0.96 (male) vs
warmth words ~0.02–0.22 (female). Under **humor** it narrows — brilliance drops to
~0.5–0.8 and humor words sit ~0.3–0.5 (roughly neutral, "funny" is a bit
male-typed), so the two bands nearly touch → the weaker `D_gb`.

### What it means

- **Both models hold the "brilliance = men" stereotype** — every cell has
  `D_gb > 0`, and the brilliance words (super smart / genius / brainiac) are
  strongly male-typed (P(male) up to 0.92–0.96 in Sonnet-warmth). **Sonnet > gpt-5.4** again.
- **The control set matters a lot — warmth ≫ humor.** With *warmth*, the effect
  is huge (D_gb ~0.71–0.84): brilliance → male, warmth → female, big gap. With
  *humor*, it's much weaker (D_gb ~0.22–0.40) and gpt-5.4's barely clears zero.
  **Why:** "funny" is itself somewhat **male-typed** (control P(male) ~0.4 under
  humor vs ~0.1 under warmth), so it competes with brilliance and shrinks the
  contrast. This exactly reproduces Storage (2020): warmth/beauty give the
  cleanest male-brilliance effect; masculine-typed controls muddy it. **Warmth is
  the cleaner instrument; humor is a conservative lower bound.**
- **Degeneracy:** humor runs had ~4/50 all-one-gender sessions (excluded);
  warmth runs had 0.

---

## PART 3 — RQ2a: SC syntax on gender (`wat_sc_gender.py` → `sc brilliance/`)

**Question.** Does the *syntactic complement* ("A is as good as B at X") make the
complement gender read as **more skilled** — on a NOVEL activity, so any effect is
pure syntax, not a domain stereotype?

**Task.** SC-scene WAT. Gender pair per trial (boys/girls or men/women, pooled to
male/female), a nonsense skill (freeching…), brilliance-vs-effort words. Conditions
`sc` (subject/complement sex randomized, split later), `ss` (equal), `noscene`.

**Scores.** `D_role` (SC only) = P(effort|subject)+P(brilliance|complement)−1,
**>0 = complement coded more skilled** (pools over sex, so a fixed gender prior
cancels). `D_gender` = P(brilliance|female)−P(brilliance|male), **+ = female
brilliant / − = male brilliant**. **Diagnostic:** a *syntactic* effect needs
`D_role>0` AND `D_gender` to **flip** with the complement sex (comp=female → +,
comp=male → −); if `D_gender` stays the same sign everywhere, it's a **fixed
gender prior**, not syntax.

### Results (n=50/cell; sc=100, ss=50, noscene=50)

| model   | D_role pooled (binomial)     | D_gender comp=female | comp=male     | ss      | noscene |
| ------- | ---------------------------- | -------------------- | ------------- | ------- | ------- |
| gpt-5.4 | −0.192 [−0.41,+0.03] (33%) | −0.782              | −0.902       | −0.533 | −0.988 |
| Sonnet  | +0.043 [−0.16,+0.25] (46%)  | −0.748              | −0.998<br /> | −0.793 | −0.956 |

### What it means

- **The syntactic effect does NOT emerge on gender.** `D_role` is ~0/negative for
  both models, and — decisively — `D_gender` **does not flip**: it's strongly
  **negative (male-brilliant) in every cell**, including when *female* is the
  complement (where syntax predicts +). So this is the **fixed "brilliance = men"
  gender prior dominating**, not a complement effect. (Consistent with the GB
  baseline: the male-brilliant prior is strong.)
- **Sharp contrast with RQ2b (novel groups, below):** the SAME syntax that
  robustly makes a *neutral* complement brilliant is **swamped by the pre-existing
  gender stereotype** — the prior overrides the syntax rather than the syntax
  flipping the prior.
- **Caveat — degeneracy:** gpt-5.4 refused to differentiate (all words → one
  gender) in **29/100** sc sessions (Sonnet 12/100); those all-one-gender dumps
  are excluded and are not missing-at-random (they skew female), biasing D_gender
  slightly more negative. So read gpt-5.4's numbers as directional.

---

## PART 4 — RQ2b: SC syntax × FAB on novel groups (`wat_sc_fab.py` → `sc fab/`)

**Question.** Does SC syntax make a NOVEL-group complement read as **brilliant**,
and is that effect **gated by FAB** (stronger for high-FAB fields)?

**Task.** SC-scene WAT. Novel group pair per trial (Zarpies/Gorps, Blickets/Daxes),
a REAL field (high- vs low-FAB, model-specific fieldset), sorted words = brilliance
vs {effort | control-set}. Conditions `sc`, `ss`, `noscene` × FAB level.

**Scores.** `D_role` = P(other|subject)+P(brilliance|complement)−1, **>0 =
complement brilliant** (pools over label → any label prior cancels). **Gating Δ**
= D_role(high-FAB) − D_role(low-FAB); **>0 (CI excludes 0) = FAB gates**. `D_group`
= slot-prior neutrality check (≈0 expected). Two instruments were run: the
brilliance-vs-**effort** (Leslie) `_eff` files and the brilliance-vs-**control**
(Storage warmth) files.

### Results (n=50/cell; sc=100, ss=100, noscene=50)

| model / instrument | D_role sc/high       | D_role sc/low        | GATING Δ (high−low)  | neutrality (ss / noscene D_group) |
| ------------------ | -------------------- | -------------------- | ---------------------- | --------------------------------- |
| Sonnet — effort   | +0.815 [+0.64,+0.96] | +0.911 [+0.79,+1.0]  | −0.096 [−0.30,+0.09] | −0.05 / +0.00                    |
| gpt-5.4 — effort  | +0.562 [+0.32,+0.77] | +0.672 [+0.47,+0.87] | −0.110 [−0.41,+0.19] | +0.01 / +0.05                     |
| gpt-5.4 — warmth  | +0.252 [+0.07,+0.43] | +0.426 [+0.22,+0.62] | −0.175 [−0.44,+0.09] | +0.04 / +0.29                     |

### What it means

- **Strong, content-neutral SC effect.** `D_role > 0` clearly in *both* FAB levels,
  for both models and both instruments — putting a neutral group in the complement
  slot makes it read as brilliant. **Sonnet's is huge** (+0.8–0.9); gpt-5.4 solid;
  the effort instrument yields a bigger effect than warmth (effort is the native
  Chestnut contrast). **No degeneracy** here (0/… everywhere) — novel groups dodge
  the gpt-5.4 refusal that plagued the gender version.
- **No FAB gating.** Every `GATING Δ` is ≈0 with a CI spanning 0 — and the point
  estimate is consistently *slightly negative* (low-FAB marginally higher). So the
  syntactic complement→brilliant effect is **FAB-independent**: it fires just as
  much on low-FAB fields. This *supports* Chestnut's Stage-1 content-neutrality and
  *argues against* the "FAB gates syntax" hypothesis.
- **Neutrality checks pass** (ss/noscene D_group ≈ 0), confirming the novel labels
  carry no prior and the SC effect is specific to the "as good as" syntax. (One
  minor blip: gpt-5.4-warmth noscene D_group +0.29 — a small slot lean, CI includes 0.)

---

## Cross-baseline takeaway

Both models carry **two human biases**: field-specific ability beliefs (high-FAB →
brilliance) **and** the gender-brilliance stereotype (brilliance → men). **Sonnet
shows both more strongly and cleanly than gpt-5.4**, whose measurements are partly
eroded by "refuse-to-differentiate" guardrail sessions.

The RQ2 syntactic studies then reveal a clean split:

- **On NEUTRAL novel groups (RQ2b):** SC syntax **robustly** makes the complement
  brilliant — content-neutral (fires regardless of FAB), **no FAB gating**. Pure
  Stage-1 syntactic effect.
- **On GENDER (RQ2a):** that same syntax is **overridden** — the strong pre-existing
  "brilliance = men" prior dominates and does **not** flip with the complement slot.

So syntax alone can *teach* a complement=brilliant asymmetry when there is no prior,
but when a strong stereotype already exists (gender), the stereotype wins over the
syntax rather than the syntax redirecting it.

*Convention reminders: D_FAB > 0 = human FAB pattern; D_gb/D_dir > 0 = brilliance→male.
All D-scores use ε=0.01 smoothing, per session, ties handled by the count-based formula.
Continuous FAB r is a field-level (n=30) correlation, distinct from the per-session D.*
