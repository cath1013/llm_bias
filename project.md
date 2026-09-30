# Project Briefing: Syntactic Framing, FAB, and Brilliance Bias in LLMs

## 1. Background concepts

Three ideas underpin this project.

**SC framing effect (syntactic-complement effect).** In "A is as good as B at X," the *complement* position (B) tends to be read as the natural reference point / more skilled, while the *subject* (A) is read as effortfully catching up. In humans this has been documented for math ("girls are as good as boys at math" → boys read as the standard). We use "SC framing" to denote the hypothesis that this syntactic structure alone can push the complement toward being seen as *more skilled* or *more brilliant*. Adapted from Chestnut & Markman ("Just as Good"). Note: their "naturally good" wording arguably operationalizes "more skilled" — needing to work harder to be as good implies being worse — so natural ability and better performance are collapsed onto one construct.

**FAB (field-specific ability beliefs).** From Leslie et al. (2015). Some academic fields are believed to require raw, innate brilliance to succeed ("high FAB"); others are believed to reward effort/empathy ("low FAB"). Because people hold a gender-brilliance stereotype, women are underrepresented in high-FAB fields.

**Brilliance bias (gender-brilliance stereotype).** From Bian, Leslie & Cimpian (2017) and Bian (2025). People implicitly associate brilliance/genius with men more than women.

## 2. Research questions

**RQ1 — Can SC framing ALONE induce a "complement = brilliant" bias, with FAB stripped out?**

- Tests whether the syntactic structure *by itself* produces the complement→brilliant association, independent of any field-specific ability beliefs.
- To remove FAB, use **novel activities** (e.g., "fleetching," "gorping") paired with either **novel groups** (e.g., "Gorp," "Modi"/"Papu") or **gender labels**.
- Uses **brilliance-vs-control words** as the outcome contrast.
- **DESIGN IS STILL OPEN.** WAT is *not* suitable here (see §4), and there is *no human baseline* for this. This needs a new design — the hardest unsolved part of the project.
- The earlier "no-scene baseline" study was a rough proxy for RQ1 (it measured complement = generally smart) but lacked proper brilliance-vs-control words.

**RQ2 — What the current WAT design can address (contingent on the FAB baseline being established first).** Two sub-versions:

- (a) Does SC syntax make the complement seem **more skilled** on a **novel activity** (i.e., can syntax alone teach a stereotype)?
- (b) Does SC syntax induce "complement = brilliant" in a **high-FAB field** but **not** a low-FAB field?
- Here **real fields carry FAB**, and the question is whether FAB *gates* the syntactic effect.

**The organizing contrast between RQ1 and RQ2:** RQ1 *removes* FAB (novel activity) to isolate pure syntax; RQ2 *keeps* FAB (real fields) to test whether FAB is necessary for the effect. Same brilliance-vs-control words are used throughout for comparability.

**Extension — Do LLMs show the human gender-brilliance bias (brilliance associated with men > women)?**

- This must be run fresh. **Confirmed gap:** Bai et al. (2025) measured gender×science, gender×career, and gender×power, but NOT gender×brilliance — their attribute words never include *brilliant / genius / super smart / brainiac*. (Verified in main text; not exhaustively checked against the SI Appendix word tables.)

## 3. Stimuli (validated from source papers)

**FAB fields — Leslie et al. (2015), Fig. 1 ordering:**

- High FAB: philosophy, physics, math, economics, music composition, computer science, astronomy, chemistry.
- Low FAB: psychology, education, anthropology, linguistics, sociology, Spanish, molecular biology, earth science.
- (Philosophy is a useful "high-FAB humanities" case that dissociates FAB from STEM.)

**Brilliance words — Bian et al. (2017) IAT, Studies 2a/2b:**

- Brilliance: super smart, brilliant, genius, brainiac.
- Control sets (pick by contrast type): physical = strong, brave, bold, tough; warmth = friendly, outgoing, kindly, chatty; humor = funny, entertaining, hilarious, witty; beauty = beautiful, elegant, graceful, pretty.
- Note from Bian: only the physical control ("strong") rivaled "brilliant" in male association; warmth/humor/beauty gave the cleaner male-brilliance effect.

## 4. Measurement notes / methodological constraints

- **WAT (Word Association Test, adapted from Bai et al.'s LLM-WAT + D-score)** requires a *binary direction* — one trait vs. another. It works when the binary comes from the **trait words** (brilliance vs. control), which is the case for the FAB baseline and RQ2.
- WAT is **unsuitable for RQ1** because a novel activity provides no directional anchor of its own; RQ1 needs a newly devised design (open problem).
- Unit of analysis is the **session**, not the word.
- D-score applies only to the WAT task.
- Use cautious statistical language appropriate to pilot power.

## 5. Things we have to do (task list)

1. **Build the FAB baseline (immediate next step).** WAT with: novel groups × high/low-FAB fields × brilliance-vs-control words. Goal: confirm whether current LLMs hold FAB bias like humans (high-FAB fields → brilliance; low-FAB fields → control). This is a **prerequisite** for RQ2.
2. **Run RQ2 once FAB baseline is established.** Implement (a) novel-activity skill version and (b) high-vs-low-FAB field version, using SC framing ("Modi is as good as Papu at [field]") with brilliance-vs-control WAT outcome.
3. **Run the gender-brilliance extension.** Men/women stimulus words × brilliance-vs-control words in a WAT, to test the male>female brilliance association. Optionally first confirm exhaustively (SI Appendix) that Bai et al. did not already run thii will s.
4. **Solve RQ1's design (open research problem).** Devise a non-WAT method to test whether SC framing alone (novel activity, no FAB) induces complement=brilliant. No human baseline exists, so this needs original design work.

## 6. Caveats for whoever picks this up

- Citations and figure/page references (esp. Leslie Fig. 1 ordering and Bian IAT stimulus tables) were read from the project PDFs but should be double-checked against the source before going into any writeup.
- Models under study in the pilot: GPT-4.5, Claude Sonnet 4.6, gpt-oss-120b, Llama Scout.
