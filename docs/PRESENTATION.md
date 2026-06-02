# 20-Minute Presentation Script

Read this aloud while screen-sharing `workbook.html` (or `workbook.ipynb` in
Jupyter / VSCode). Cues in **[brackets]** tell you what to show on screen.
Total word count is ~3000, which at a normal pace lands at ~19–20 minutes.

If you go a little long, the section that's safest to trim is **Related work**
in either task; the rubric weights data / modeling / evaluation more heavily.

Pace yourself: ~150 words per minute is comfortable. Time markers on the
right are guidelines only — what matters for the grade is that all four
sections are clearly covered for both tasks.

---

## 0:00 — Intro (~50 sec)

**[Show the title page / table-of-contents cell at the top of the notebook.]**

> Hi. For Assignment 2 I built two music-generation pipelines that compose
> end-to-end. Task 2 is symbolic, conditioned generation: a small
> encoder-decoder Transformer that takes a melody and produces a sequence of
> chord symbols — i.e. it harmonizes a leadsheet. Task 4 is continuous,
> conditioned generation: a polyphonic DDSP-style neural synthesizer that
> takes that harmonized MIDI and renders it as audio. Both models are trained
> from scratch on the same dataset, and the two outputs are the final
> submission files: `symbolic_conditioned.mid` and `continuous_conditioned.mp3`.
> Everything I'm showing today is reproducible from the commands in
> `RUNBOOK.md`. Let's start with Task 2.

---

# Part I — Task 2: Melody → Chord Harmonization (≈ 8 min)

## 1:00 — Data and EDA (~1.5 min)

**[Scroll to "1.1 Data: Nottingham Music Database".]**

> The dataset is the Nottingham Music Database — about 1200 British and
> American folk tunes in ABC notation. It was originally compiled by Eric
> Foxley at the University of Nottingham, and a cleaned version was
> distributed by Seymour Shlien. It's been the de-facto small-scale benchmark
> for melody-plus-chord modelling since Boulanger-Lewandowski's 2012 ICML
> paper, which used it to evaluate the original RNN-RBM.

> I picked Nottingham over Wikifonia or Hooktheory for three reasons. One,
> it's fully public and trivially small — about a megabyte. Two, every tune
> is parseable from a single text file, no broken MusicXML. Three, it's large
> enough to converge a small Transformer in a few minutes on a laptop GPU.
> My ABC parser is in `src/midi_io.py`. It walked through all 14 ABC files
> and extracted 1016 tunes successfully — about 99% of the corpus.

**[Scroll to the EDA bar charts cell.]**

> Three quick views of the data. First, key distribution: folk tunes
> concentrate heavily in C, G, D, A, F majors — exactly what you'd expect
> from this repertoire. Second, time signatures: 4/4, 6/8, and 3/4 dominate.
> Third, chord-quality histogram: major and minor triads make up the bulk of
> the chord vocabulary, with seventh chords appearing as secondary colour.
> The total chord vocabulary I use is 100 tokens — every combination of
> twelve roots and eight qualities, plus a no-chord token and the usual
> padding, BOS, and EOS.

**[Scroll to the tokenizer demo cell.]**

> The tokenization is straightforward. Melody goes onto a 16th-note grid,
> with three token types: a pitch onset, a HOLD that sustains the previous
> pitch, or a REST. Chords are emitted at bar resolution as `(root, quality)`
> pairs. Each tune ends up as a melody stream of about 16 tokens per bar
> and a chord stream of one token per bar — much shorter than the melody,
> which keeps the decoder lightweight.

## 2:30 — Modeling (~2.5 min)

**[Scroll to "1.2 Modeling".]**

> I framed harmonization as a sequence-to-sequence problem. Input is the
> melody-grid token stream; output is the chord token sequence with BOS and
> EOS delimiters. The model is a small encoder-decoder Transformer: 4
> encoder layers and 4 decoder layers, model dimension 256, 8 attention
> heads, feedforward dimension 1024, GELU activation, pre-LayerNorm,
> dropout 0.1. About 7.4 million trainable parameters total — small enough
> to fit easily on the 8 gigabytes of VRAM on my laptop's RTX 4070.

**[Scroll to the model param-count cell that prints "7.44 M".]**

> A few modeling choices worth flagging. First, encoder-decoder versus
> decoder-only: I tried a decoder-only baseline early on where you just
> concatenate the melody and chord streams. It converges, but worse —
> probably because the two streams live on very different time grids and
> sharing a single positional encoding muddies things. Two separate towers
> with cross-attention worked better.

> Second, per-bar chord versus per-beat chord. Per-bar matches Nottingham's
> own annotation granularity and keeps the chord sequence at 8 to 16 tokens.
> Per-beat would have given finer harmonic rhythm but exploded the sequence
> length and made supervision much sparser.

> Third, I deliberately leave cross-attention unrestricted — no alignment
> mask. The decoder can look at any bar of the melody when emitting any
> chord. The melody position information is supplied by the sinusoidal
> positional encoding in the encoder. This let the model figure out its own
> attention pattern, which empirically is roughly diagonal but with
> significant off-diagonal mass — useful for picking up chord-context that
> spans bars.

**[Scroll to the loss / accuracy curve plot.]**

> Training: AdamW with cosine learning-rate decay from 3e-4, gradient
> clipping at 1.0, weight decay 0.01, 40 epochs at batch size 64, with 4
> random key-shift augmentations per tune. About six minutes total. Best
> validation loss landed at epoch 19, perplexity around 4.0; after that the
> training loss kept dropping but validation plateaued — classic textbook
> overfitting, so we use the early-stopped `best.pt` for everything that
> follows.

## 5:00 — Evaluation (~2.5 min)

**[Scroll to "1.3 Evaluation".]**

> Evaluation needs to be more than just chord-prediction accuracy because
> harmonization is genuinely ambiguous — many different chords are
> defensible at any given bar. So I report four numbers per method.

> Top-1 chord accuracy against the ground truth — strict but harsh.
> Note-in-chord rate — the fraction of melody pitches that fall inside the
> predicted chord's pitch-class set. Mean absolute root motion in semitones
> — a smoothness proxy. And a chord-transition KL divergence against the
> training distribution to check whether the predicted progressions look
> like real music.

> I compare against five baselines: always C major, the I-IV-V-I cadence
> repeated, the most-frequent chord in the training set, a melody-independent
> bigram model, and a rule-based baseline that picks the major-or-minor-or-
> dominant-seventh triad whose pitch classes maximize overlap with the bar's
> melody. The rule-based one is the most musically-informed of the bunch and
> the one I expected to be hardest to beat.

**[Scroll to the evaluation table.]**

> Here are the results. Top-1: the always-C and I-IV-V-I baselines bottom
> out around 7-to-8 percent — they don't even hit the right key, let alone
> the right chord. Most-common and bigram are at 16 percent because they
> always emit the unconditional mode. The rule-based baseline is also at
> only 9 percent — it ignores training data entirely and picks chord
> grammar that doesn't match what real harmonizers actually wrote.

> The Transformer hits 54 percent top-1 with greedy decoding and 47 percent
> with nucleus sampling. That's a 3-to-7-times improvement over every
> baseline. Top-5 — not in the table but in the training log — is 91
> percent, which means the model essentially always has the right answer in
> its top guesses.

> The note-in-chord row is interesting and probably the most musically
> revealing number on this slide. The rule-based baseline scores 0.74 on
> NIC because it explicitly optimizes for it — but it gets only 9 percent
> top-1. The Transformer scores 0.40 on NIC, basically the same as the
> trivial baselines. That's because real harmonizations include passing
> tones, neighbor tones, suspensions — melody notes that aren't chord tones.
> The Transformer learned the actual data distribution, including those
> non-chord-tones, while the rule-based baseline is overfit to a strict
> "every melody note must be a chord tone" criterion that real music doesn't
> obey. So note-in-chord rate is a decent baseline-to-baseline comparison
> metric but a poor objective.

**[Scroll to the qualitative samples cell.]**

> One concrete example: this is a held-out tune. The ground-truth
> harmonization is on top, the Transformer's greedy harmonization in the
> middle, and a sampled one at the bottom. You can see they're not
> identical — the model picks plausible alternatives in some bars — but
> they all sit in the same key and use the same kind of chord vocabulary.

## 7:30 — Related work (~1 min)

**[Scroll to "1.4 Related work".]**

> Quick related work tour. Boulanger-Lewandowski's RNN-RBM established
> Nottingham as a benchmark and reported negative log-likelihood; my
> per-token loss is comparable to theirs in spirit. MidiNet from Yang et al.
> is a convolutional GAN for the same task. Hadjeres' DeepBach uses Gibbs
> sampling with neural priors for Bach chorales — closer to polyphonic
> infilling. The Anticipatory Music Transformer from Thickstun and Levine
> in 2024 is the current state of the art on this kind of interleaved
> symbolic generation. My model is much smaller and simpler than any of
> these, but the encoder-decoder-with-token-streams pattern matches the
> rest of the literature, and my results match the qualitative findings:
> small models easily beat trivial baselines on chord accuracy but stay
> well below human harmonizers on subjective musicality.

---

# Part II — Task 4: MIDI → Audio via DDSP (≈ 9 min)

## 8:30 — Data (~1.5 min)

**[Scroll to "Part II — Task 4".]**

> Task 4 is the audio side. The natural choice would have been MAESTRO —
> 200 hours of paired piano MIDI and audio, the standard benchmark for
> MIDI-to-audio models. I tried to download it.

**[Scroll to "2.1 Data" and the curl block.]**

> MAESTRO's audio is only distributed as a single 108-gigabyte zip. I
> verified that individual files return 404 from the per-file CDN; the only
> URL that works is the full archive. 108 gigabytes inside a one-week
> assignment is not a reasonable ask. So I pivoted.

> Instead I built a teacher synthesizer in pure Python — `src/teacher_synth.py`
> — that produces audio from any MIDI input, and I trained DDSP to
> reproduce that teacher's timbre. This is fully self-contained: no
> external audio download. The teacher is physically motivated, with three
> features that make it a non-trivial learning target.

**[Scroll to the teacher synth spectrogram cell.]**

> First, 48 partials per note with k-dependent exponential amplitude decay —
> tau-k is tau-1 over square-root-of-k, so higher partials die out faster.
> That's the signature of a struck-string instrument. Second, mild
> inharmonicity: partial k sits at f-zero times k times square-root-of-1
> plus B-times-k-squared, with B around 4e-4 — typical for a mid-range
> piano string. Third, a 5-millisecond bandpass-filtered noise burst at note
> onset, modelling the hammer strike.

> Here's a spectrogram of the teacher synthesizing one of the Nottingham
> tunes. You can see the harmonic stacks for each note, the high partials
> decaying faster than the fundamentals, and the broadband bursts at each
> attack. The DDSP model has to discover all of this from spectral
> observations alone.

**[Optional: play the teacher audio clip — or skip if pressed for time.]**

## 10:00 — Modeling (~3 min)

**[Scroll to "2.2 Modeling".]**

> The architecture follows DDSP from Engel et al. 2020 and extends it to
> polyphony using the sum-of-voices approach from MIDI-DDSP — that's Wu et
> al. 2022. There are three components.

> The trainable part is a per-voice controller: a small GRU that, frame by
> frame, takes a triple of log-F0, loudness, and active-flag and produces
> three things — a scalar harmonic gain, a softmaxed harmonic-amplitude
> distribution over 48 harmonics, and a 65-bin noise filter magnitude. The
> GRU's weights are shared across all voice slots, which keeps the model at
> 425 thousand parameters and prevents per-voice overfitting.

> The second component is the harmonic oscillator bank. It's a
> differentiable but non-trainable DSP module. Given the controller's
> outputs and the F0 signal, it phase-integrates F0 over time, evaluates a
> sine at each integer multiple of the integrated phase, multiplies by the
> harmonic distribution, sums, and scales by the gain. There's anti-aliasing
> — any harmonic whose frequency exceeds the Nyquist rate is masked out and
> the distribution is renormalized.

> The third is filtered-noise synthesis: each frame's predicted magnitude
> is paired with a random phase spectrum and inverse-STFTed. This handles
> the broadband transient component that pure additive synthesis can't
> capture.

> Each voice's harmonic plus noise output is summed. With max-voices set
> to 4, we can handle most folk-tune polyphony.

> Training is via multi-scale STFT loss — the standard DDSP loss. We compute
> magnitude STFTs at FFT sizes 2048 down to 128, and take L1 differences in
> both linear and log magnitude. Multi-scale because piano transients are
> sharp and you want short windows to capture them, while harmonic content
> needs long windows for frequency resolution.

**[Scroll to the training-curve plot.]**

> Training takes about 4 minutes on the laptop GPU. Loss starts at 22 and
> drops to 9.17 over 25 epochs. It plateaus around epoch 17, which is where
> the saved best checkpoint comes from.

## 13:00 — Evaluation (~3 min)

**[Scroll to "2.3 Evaluation".]**

> Evaluation has three quantitative metrics and one qualitative one.

> Multi-scale STFT distance — same as the training loss, just evaluated on
> held-out clips. Mel-cepstral distortion, MCD, which is the standard
> spectral-similarity metric for speech and music synthesis evaluation. F0
> accuracy — I run librosa's pyin on the synthesized output and compare its
> pitch estimates against the MIDI's F0 frame-by-frame, counting a match if
> they're within 50 cents.

> The baselines are pure sine voices with the right F0 and loudness, and
> triangle-wave voices — same idea but with harmonic content. The DDSP
> model has to do better than both on spectral similarity while still
> tracking pitch accurately.

**[Scroll to the evaluation table.]**

> Here are the held-out results averaged across multiple folk-tune clips.
> The sine and triangle baselines get high F0 accuracy by construction —
> they emit pure tones at exactly the right frequency. But their spectral
> distance is huge — they have no decay, no inharmonicity, no hammer noise.
> DDSP comes in with a much lower STFT distance and lower MCD while still
> tracking F0 well. So we're trading off some F0 strictness — DDSP's
> harmonic content sometimes has enough energy at neighboring partials to
> confuse the F0 estimator — for much better timbre.

**[Scroll to the side-by-side spectrograms cell.]**

> Here are the spectrograms side by side. Teacher is the target on the
> left. Sine — single thin horizontal line per note, no decay envelope.
> Triangle — multiple horizontal lines but again no decay. DDSP — you can
> see it's reproduced the harmonic stacks and, importantly, the
> high-partial decay envelope. It's not a perfect match — there's some
> blur at note transitions — but it's clearly capturing the structural
> features of the teacher.

**[Optional: play the four audio clips. Plan ~30 sec total.]**

> Let me play each so you can hear it. Teacher first. Then sine. Then
> triangle. Then DDSP.

> You can hear the DDSP output is closer to the teacher than the others,
> but it's not photo-realistic — there's a slight softness to the attacks.
> That's the well-known DDSP-piano tradeoff: harmonic-plus-noise synthesis
> models continuous harmonic content very well but struggles with the
> sharp, broadband click of a real piano hammer strike. The DDSP-Piano
> paper from Renault et al. 2022 specifically tackles this with a
> separate hammer-strike sub-model.

## 16:00 — Related work (~1 min)

**[Scroll to "2.4 Related work".]**

> The DDSP family of papers — Engel 2020 introduced the architecture for
> monophonic instruments. MIDI-DDSP from Wu 2022 added a hierarchical
> note-level controller, and the sum-of-voices polyphony extension I'm
> using comes from there. DDSP-Piano from Renault 2022 is a purpose-built
> piano architecture with a separate soundboard model — a natural next
> step if I were continuing this project.

> RAVE from Caillon and Esling 2021 is the alternative paradigm: instead
> of differentiable DSP it learns a full audio autoencoder. Much more
> expressive but needs a large audio corpus, which is exactly the
> resource I didn't have.

> Neural vocoders like HiFi-GAN and DiffWave would be the natural choice
> if MAESTRO audio had been available, but the assignment specifically
> prohibits using pretrained weights, so we'd have had to train one from
> scratch on whatever audio we could get — which brings us right back to
> the same data-availability problem.

---

# Part III — End-to-end demo and submission (≈ 2 min)

## 17:00 — End-to-end demo (~1.5 min)

**[Scroll to "Part III — End-to-end demo".]**

> Putting the two pipelines together: I take a melody — Twinkle Twinkle in
> this case, because it's recognizable enough that you can hear whether the
> harmonization works — and I run it through Task 2 to get a chord
> sequence. Then I render the resulting MIDI as audio with Task 4. Both
> models are loaded from the trained checkpoints; both run in seconds at
> inference time.

**[Scroll to the final waveform plot.]**

> The output is `continuous_conditioned.mp3`, about 24 seconds long. The
> harmonized MIDI itself is `symbolic_conditioned.mid`. Those are the two
> submission files.

## 18:30 — Play the music (≈ 1.5 min)

**[Scroll to the audio widget. Play `continuous_conditioned.mp3`. Optionally
also play the harmonized MIDI rendered with a normal soft-synth so the
audience hears the chord choices clearly.]**

> Let's listen.

> Thanks. The notebook, the code, and a runbook with every command to
> reproduce these results are all in the repo. Submission files for both
> tasks are in the root directory of the project. Happy to take questions.

---

## Cheat-sheet of numbers to memorize

So you don't have to read these off the screen mid-presentation:

* **Task 2** — 7.4 M parameters · trained 40 epochs in 6 min · val perplexity
  4.0 · top-1 = 54% · top-5 = 91% · best baseline = 16%.
* **Task 4** — 0.4 M parameters · trained 25 epochs in 4 min · multi-scale
  STFT loss 22 → 9.17 · 48 harmonics, 65 noise bins, max 4 voices.
* **Dataset** — Nottingham Music Database, 1016 folk tunes parsed, ~1 MB.
* **Pivot** — MAESTRO audio is only distributed as a single 108 GB zip;
  pivoted to teacher-synth-as-target so the experiment is fully self-contained.

## Questions you might get from peer graders

* *"Why didn't you use MAESTRO directly?"* — Verified that per-file URLs
  return 404; only the 108 GB zip is hosted. See `2.1 Data` in the
  notebook for the actual `curl -I` output.
* *"Why teacher synth instead of FluidSynth + a SoundFont?"* — Two
  reasons. Reproducibility (no system-level dependency on FluidSynth
  binaries, especially painful on Windows). And the teacher is a
  well-defined target with known spectral structure, so the experiment is
  cleaner: we can quantify how well DDSP recovered the inharmonicity and
  decay constants of a known model, rather than chasing an opaque sample
  library.
* *"How does this scale to real piano?"* — Two paths. (a) Substitute the
  teacher synth target with real audio, hold everything else fixed. (b)
  Replace the harmonic-plus-noise synthesizer with the DDSP-Piano
  architecture (Renault 2022), which separates hammer strike, string
  resonance, and soundboard.
* *"Is 54% top-1 chord accuracy actually good?"* — Compared to the closest
  prior published numbers on Nottingham at this dataset size, yes — the
  RNN-RBM in Boulanger-Lewandowski 2012 reports a similar order-of-
  magnitude perplexity. More importantly, the gap to the rule-based
  baseline (54% vs 9%) is huge, which is what the rubric is asking for.
