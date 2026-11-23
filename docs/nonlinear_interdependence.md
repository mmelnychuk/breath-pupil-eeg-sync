# Nearest-neighbour coupling of breathing, pupil diameter, and electroencephalography

This is an offline analysis. It does not change how a session is recorded. It reads files that already share Unix time (seconds since 1970-01-01 00:00:00 UTC) and asks how much one signal's recent pattern predicts another's.

Three signals are used.

* **Breathing**, the nasal-pressure column of `breathing.csv`.
* **Pupil diameter**, from a Pupil Core export. It is the observable used here as a stand-in for locus coeruleus activity, following Melnychuk et al. (2018).
* **Electroencephalography** from `eeg.csv`, the fourteen original Emotiv EPOC channels AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4. The headset has no Fz. The default average is the frontal set AF3, AF4, F3, F4.

The scientific question comes from two papers. Melnychuk et al. (2018), Psychophysiology, doi 10.1111/psyp.13091, equation 2, couple a respiratory oscillator, a locus-coeruleus oscillator, and an attentional oscillator. The printed equation puts the locus-coeruleus coefficient on the respiratory line. The author confirmed that this is a typographical error: respiration has its own coefficient. There is no direct breathing–attention term. Melnychuk et al. (2021), Brain Sciences, doi 10.3390/brainsci11101324, later found linear multivariate Granger causality in all six directions between breathing, an electroencephalography electrode, and pupil diameter, which is consistent with a direct breathing–cortex route (for example through olfaction) as well as the route through arousal. The conditional index below is there to separate those two accounts.

## Preprocessing

Breathing, pupil diameter, and the electroencephalography envelopes are placed on one evenly spaced grid. The default rate is 16 Hz. Anything from about 10 to 20 Hz is in the range this was written for: fast enough to follow a breath, slow enough that a few minutes of recording stay a manageable number of states.

Breathing times are Unix seconds. A file whose time column is `time.time()` multiplied by 1000 (values larger than 10^11) is divided by 1000, which is the `Sniff_polar_display.py` convention. Rows that carry a trigger label and an empty pressure are ignored.

Pupil diameter prefers the millimetre column `diameter_3d` over the pixel diameter. A sample is treated as a blink when its confidence is below 0.6 (the same threshold the live view uses), when the diameter is missing or not positive, or when it sits within 50 ms of such a sample. Gaps of up to half a second are filled by straight-line interpolation. Longer gaps stay missing, and the analysis keeps the longest stretch on which all three signals are present. If the export is still on Pupil Capture's own clock, pass `--pupil-offset` equal to the offset the hub prints at startup (pupil time minus Unix time). Unix time is then pupil time minus that offset. The hub plots pupil diameter during a session and does not write a continuous diameter file; a Pupil Core `pupil_positions.csv` has to be copied in, or passed with `--pupil`.

Electroencephalography is not analysed as the raw trace. Each selected channel is band-passed, at the file's own sample rate, into:

| Band | Hertz |
| --- | --- |
| delta | 0.5–4 |
| theta | 4–8 |
| alpha | 8–13 |
| beta | 13–30 |
| low gamma | 30–45 |

A band whose upper edge is above the Nyquist frequency, or that the record is too short to filter, is skipped and named in the run log. The original EPOC streams at 128 Hz, which covers low gamma. A short simulated session written at 32 Hz does not.

The analytic signal is the Hilbert transform. Instantaneous power is the square of its amplitude. That power is low-pass filtered at 2 Hz (zero phase) and the first and last half-second are dropped so the filter's warm-up is not treated as brain activity. Powers are averaged across the selected channels, then resampled onto the common grid. Every ordered ratio of those powers is kept as its own candidate attentional signal: delta/alpha, alpha/delta, theta/beta, and the rest. No ratio is given a special status in the statistics. The denominator is floored at 0.1% of its median so a near-zero power does not explode the ratio.

## Delay embedding

Each signal is rescaled to mean 0 and standard deviation 1, then delayed. The state at time t is the vector

    (x(t), x(t − τ), x(t − 2τ), …, x(t − (m − 1)τ))

τ is the delay in samples and m is the dimension. The delay is the first local minimum of the mutual information between the signal and a lagged copy of itself (Fraser and Swinney, 1986). If the curve never turns upward, the delay is the first lag at which the mutual information has fallen below 1/e of its value at one sample. The dimension is the smallest m whose false-nearest-neighbour fraction is at most 0.1 (Kennel, Brown, and Abarbanel, 1992). A neighbour is false when stretching the vector by one more delay multiplies its distance by more than 15, or when that stretched distance exceeds twice the standard deviation of the signal.

The Theiler window (Theiler, 1986) is the number of samples around a state that are forbidden as neighbours. The default is the longest embedding window in the triplet, (m − 1)τ, and at least one sample. It stops a state being paired with the overlapping vectors built from the same samples. For the oscillator checks the window is one respiratory cycle, which is longer than the embedding window.

On the simulated oscillators the delay is set to a quarter of a respiratory cycle and the dimension to 3, rather than estimated, so the ground-truth checks are not entangled with the delay picker. Real recordings use the estimates above.

## Bivariate indices

These four indices are the state-space nearest-neighbour family. Let x be the target state and y the source state. Let r(n, j) be the time indices of the k nearest neighbours of x(n), and s(n, j) the time indices of the k nearest neighbours of y(n). Neighbours inside the Theiler window are excluded. Squared Euclidean distance is used throughout. k defaults to 10 on a recording and to 8 in the oscillator checks. Quian Quiroga, Kraskov, Kreuz, and Grassberger (2002) give S, H, and N. Andrzejak, Kraskov, Stögbauer, Mormann, and Lehnertz (2003) give M. Arnhold, Grassberger, Lehnertz, and Elger (1999) introduced the underlying radii.

    R_k(X)     = average squared distance from x(n) to its own k nearest neighbours
    R_k(X | Y) = the same average, but using the time indices of y's neighbours
    R(X)       = average squared distance from x(n) to every state outside the Theiler window

The reported indices are averages over states n:

    similarity S                  = average of  R_k(X) / R_k(X | Y)
    interdependence H             = average of  ln R(X) − ln R_k(X | Y)
    normalised interdependence N  = average of  (R(X) − R_k(X | Y)) / R(X)
    Andrzejak interdependence M   = average of  (R(X) − R_k(X | Y)) / (R(X) − R_k(X))

H uses the natural logarithm. S, N, and M are near 0 when the source tells you nothing about the target, and near 1 when the source's neighbours are also the target's neighbours. H is 0 when the source is uninformative and grows as those neighbours get closer; it is not bounded by 1. M is the one plotted by default, because the denominator removes the part of the score that is just the target's own smoothness.

The arrow in the output, "breathing → pupil diameter", means the index of the pupil given breathing: breathing's neighbours are being used to describe the pupil. A high score means the breathing state predicts the pupil state. It is not, by itself, a statement about which physiological process drives the other. Exchanging the two signals gives the opposite arrow, and the two numbers need not match.

## Conditional indices

A high bivariate score between breathing and an attentional signal can happen because both are coupled to the locus coeruleus, with no direct breathing–attention term. The conditional indices ask whether the candidate source still helps once the third signal has been taken into account.

Fix a target Y, a candidate source X, and a conditioner Z (the signal that is neither X nor Y). For each state, take a pool of P nearest neighbours in Z. P defaults to five times k. Two sets of k states are then cut from that pool:

* the k members closest to the current state in Z (Z's ordinary neighbours)
* the k members closest to the current state in X (the pool re-ranked by the candidate source)

Write R(Y | Z) and R(Y | X; Z) for the mean squared distance in Y to those two sets, and R_k(Y) for the distance to Y's own neighbours. The conditional indices are zero when re-ranking by X does not move the target, and positive when it pulls the target closer:

    conditional similarity S                  = average of  R(Y | Z) / R(Y | X; Z) − 1
    conditional interdependence H             = average of  ln R(Y | Z) − ln R(Y | X; Z)
    conditional normalised interdependence N  = average of  (R(Y | Z) − R(Y | X; Z)) / R(Y | Z)
    conditional Andrzejak M                   = average of  (R(Y | Z) − R(Y | X; Z)) / (R(Y | Z) − R_k(Y))

Conditional M is the one used to decide whether a direct link was detected. Re-ranking inside a pool, rather than concatenating X onto the distance, avoids giving an irrelevant extra dimension the chance to scramble the neighbours.

There is a second conditional score, a nearest-neighbour forecast. Neighbours are sought in the joint delay embedding of the target and the conditioner (each signal scaled by 1/sqrt(m) so a higher dimension does not dominate). The baseline forecast of the target a short time ahead is the mean of the futures of the k closest of those neighbours. The refined forecast re-ranks the same pool by distance in the source. The prediction gain is

    1 − (refined mean squared error) / (baseline mean squared error)

and the reported value is the largest gain over horizons of 0.1, 0.2, 0.4, 0.8, and 1.2 seconds. The same horizons are searched for every surrogate, so picking the best horizon does not by itself make a result significant. This is the form "neighbours in a joint embedding of the target and the third signal, with and without the candidate source". On these slow oscillators it is less sensitive than conditional M; both are written to the table.

## Joint index

For each choice of target, the other two signals are the drivers. Andrzejak M is computed from each driver alone and from the two concatenated (again scaled by 1/sqrt(m)). The joint gain is the two-driver score minus the better single-driver score. Positive means the pair matches the target more closely than the better member of the pair. The surrogate for this row shifts both drivers by the same amount, so the relationship between the drivers is kept and only their alignment with the target is broken.

## Partial cross mapping

Partial cross mapping (Leng, Ma, Kurths, et al., 2020, Nature Communications, doi 10.1038/s41467-020-16238-0) is run on the same triplets so the conditional nearest-neighbour index can be compared with a method built to remove indirect links.

Mutual cross mapping follows Sugihara's convergent cross mapping. If a cause leaves a trace in an effect, the effect's delay vectors have neighbours whose times carry similar values of the cause. Predicting the cause from the effect's neighbours is evidence that the cause influences the effect. The arrow runs from the series being predicted (the cause) to the series whose neighbours are used (the effect). That is the same arrow direction as the interdependence tables, but the geometry is different: cross mapping reads the cause out of the effect, whereas Andrzejak M asks whether the cause's neighbours are close in the effect.

The cross-map prediction uses m + 1 neighbours, the smallest simplex in that dimension, with Sugihara weights: the nearest neighbour has weight 1 and the others fall off as exp(−distance / nearest distance). A lag of up to two seconds (one respiratory cycle in the oscillator checks, capped the same way) is searched, and the lag with the largest absolute correlation is kept. The cross-map correlation ρ_C is that absolute correlation. It still lights up when the influence is only indirect, because the intermediate signal's trace is copied onward.

Partial cross mapping removes that indirect trace. For a candidate cause X, an effect Y, and a third signal Z:

1. Cross-map Z from Y at the best lag. The resulting series is the trace of Z visible in Y.
2. Cross-map X from that trace. This is the part of X that Y knows only because it knows Z.
3. Cross-map X from Y itself. This is everything Y knows about X.
4. The direct-link index ρ_D is the absolute partial correlation between X and the prediction in step 3, after the prediction in step 2 has been removed:

       ρ_D = | (r_xy − r_xz r_yz) / sqrt((1 − r_xz²) (1 − r_yz²)) |

ρ_D is missing, not zero, when the third signal already explains essentially all of the variance. Leng et al. exclude strongly synchronised systems for this reason: on a noise-free limit cycle every phase is a function of every other phase, and partialling is undefined. The oscillator checks therefore add phase noise (see below). A missing direct-link index is reported as not detected.

## Surrogates and multiple comparisons

A link is marked detected only when two things are both true: the index is positive, and its one-sided p-value is at most alpha (default 0.05). Positivity matters for the conditional indices. A time-shifted source is actively worse than the conditioner's own neighbours, so the surrogate distribution sits below zero. A negative index that merely beats that baseline is not evidence that the source helped.

The p-value is

    p = (1 + number of surrogates ≥ the data) / (1 + number of surrogates)

Three null series are implemented. The default, and the one used in the oscillator checks, is the time shift: the candidate source is rotated by at least 15% of the record and at most half of it. The waveform is unchanged; the alignment with the other signals is not. A shift of only a fraction of a cycle does not destroy a phase lock (it is just a new constant phase offset), which is why the minimum shift is a substantial fraction of the record rather than a single sample.

Phase randomisation keeps the Fourier amplitudes and draws new phases, so the power spectrum stays the same. The iteratively amplitude-adjusted Fourier transform (Schreiber and Schmitz, 1996, 2000) keeps the power spectrum exactly and the set of values almost exactly. It finishes on a spectrum match. Both are available with `--surrogate phase-random` and `--surrogate iaaft`. For a nearly sinusoidal lock, a spectrum-preserving surrogate is still almost a sinusoid of the same frequency, and two sinusoids of one frequency remain strongly interdependent at any phase offset. Time shifts that are long compared with the phase-diffusion time are the null used for the oscillator model. Spectrum-preserving surrogates are the better default for broadband electroencephalography, where the autocorrelation is not a single cycle.

`--correction` applies a correction inside each measure, across every direction and every attentional signal, and not across different measures (S and M are not one family). `bonferroni` multiplies p-values by the number of tests. `fdr` is the Benjamini–Hochberg false-discovery rate. The default on the command line is no correction, because a single planned comparison should not be charged for a search the user did not run. With every band and every ratio included, turn on `--correction fdr`.

## The oscillator model

The simulated phases obey, with one coefficient per directed link,

    dφ_A / dt  = ω_A  + ε_{LC→A} sin(φ_LC − φ_A) + ε_{R→A} sin(φ_R − φ_A) + ξ_A
    dφ_LC / dt = ω_LC + ε_{A→LC} sin(φ_A − φ_LC) + ε_{R→LC} sin(φ_R − φ_LC) + ξ_LC
    dφ_R / dt  = ω_R  + ε_{LC→R} sin(φ_LC − φ_R) + ε_{A→R} sin(φ_A − φ_R) + ξ_R

ξ is Gaussian white noise and the integration is Euler–Maruyama with a 5 ms step. Figure 5 of the 2018 paper used a single coefficient of 0.33 for a stable limit cycle at respiratory frequencies 0.35 Hz and 0.25 Hz, and 0.45 at 0.12 Hz. The author describes these as values found by searching the parameter space, not as physiological constants. The published scenario puts that coefficient on the four model links (locus coeruleus ↔ attention and locus coeruleus ↔ respiration) and leaves both breathing–attention coefficients at zero. The direct scenario sets those two coefficients to the same reference value. The uncoupled scenario sets every coefficient to zero.

Natural frequencies sit near the respiratory frequency: the locus coeruleus is 0.03 Hz slower and attention is 0.04 Hz faster. Uncoupled oscillators then do not share a period by accident, and the reference coefficients are still large enough to phase-lock the detuning (the frequency difference, times 2π, is below the coefficient).

The observables are

    breathing       = sin(φ_R)
    pupil diameter  = sin(φ_LC) + 0.3 sin(2 φ_LC)
    attention       = sin(φ_A)

plus independent measurement noise of standard deviation 0.05. The pupil series is in arbitrary units, not millimetres. The second harmonic makes it asymmetric, closer to hippus than a pure sine.

Phase noise has intensity 0.55 radians per square-root second. It is not a value from the 2018 paper. Without it the three phases collapse onto one limit cycle, every phase is a function of every other phase, and no conditional method can tell a direct link from a mediated one. Leng et al. make the same exclusion for strongly synchronised systems. 0.55 is high enough that the conditional index can see the difference, and low enough that the published links still phase-lock. The checks use about 50 respiratory cycles (at least 180 seconds) at 10 Hz.

## What the simulations showed

Every number in this section is a proportion across seeds 0 through 19. No seed was removed after looking at the result. On one seed, "detected" means the index is positive and the one-sided time-shift p-value is at most 0.05. With 19 surrogates that smallest p-value is 0.05, so a detection means the data beat every surrogate. Phase noise is 0.55. Scenario records cover about 50 respiratory cycles (at least 180 seconds). The sweep uses 120 seconds at 0.25 Hz.

A coefficient is the point where a link "starts detecting" only when the detection rate across seeds first reaches 80%. That bar was fixed before the sweep was scored. If a link never reaches it, that is reported. The rate at every coefficient is listed either way.

The tables are filled from `tests/test_nlid_scenarios.py`. A rate well below 1 on a link that is really there, or well above 0 on a link that is absent, is a miss or a false positive. It is not a seed that was discarded.

### Published model, no direct breathing–attention link

Pairwise Andrzejak M does see breathing ↔ attention. That is the indirect route through the locus coeruleus. Conditional Andrzejak M given pupil diameter does not: 0 of 20 seeds at every frequency, both directions, with a mean index about −0.11 to −0.13. The four links that are in the equation are detected on 18 to 20 of 20 seeds, pairwise and conditional.

Partial cross mapping does not remove that indirect route. Its direct-link index calls breathing → attention given pupil present on 20/20, 17/20, and 16/20 seeds at 0.12, 0.25, and 0.35 Hz (mean index about 0.19 to 0.30). The reverse direction is 20/20, 17/20, and 15/20. The forecast (prediction gain) agrees with conditional M on the indirect link (0 to 2 of 20 seeds) but is weak on links that are really there, often under half the seeds at 0.25 and 0.35 Hz.

| Respiratory frequency | Pairwise M, breathing → attention | Conditional M, breathing → attention given pupil | Conditional M, attention → breathing given pupil | Partial cross mapping, breathing → attention given pupil |
| --- | --- | --- | --- | --- |
| 0.12 Hz (coefficient 0.45) | 20/20, mean 0.346 | 0/20, mean −0.116 | 0/20, mean −0.117 | 20/20, mean 0.297 |
| 0.25 Hz (coefficient 0.33) | 18/20, mean 0.149 | 0/20, mean −0.120 | 0/20, mean −0.130 | 17/20, mean 0.190 |
| 0.35 Hz (coefficient 0.33) | 18/20, mean 0.156 | 0/20, mean −0.109 | 0/20, mean −0.114 | 16/20, mean 0.189 |

### Direct breathing–attention link

Conditional M detects the added link on most seeds. Pairwise M detects it on every seed. It is not perfect at coefficient 0.33: breathing → attention given pupil is 17/20 at 0.25 Hz. At 0.12 Hz, where the coefficient is 0.45, both directions are 20/20.

Adding the direct link makes original links harder to see conditionally. Attention → pupil diameter given breathing, which is a real link, falls to 10/20, 9/20, and 8/20 at 0.12, 0.25, and 0.35 Hz, and the mean index sits on zero. Pupil diameter → attention given breathing is 12/20, 12/20, and 10/20. The signals have become redundant. Partial cross mapping calls every direction, including the direct one, on 20/20 seeds. It does not have this redundancy failure, and it also does not reject the indirect route in the published model.

| Respiratory frequency | Conditional M, breathing → attention given pupil | Conditional M, attention → breathing given pupil | Prediction gain, breathing → attention given pupil |
| --- | --- | --- | --- |
| 0.12 Hz (0.45) | 20/20, mean 0.074 | 20/20, mean 0.067 | 16/20, mean 0.029 |
| 0.25 Hz (0.33) | 17/20, mean 0.071 | 19/20, mean 0.082 | 8/20, mean 0.024 |
| 0.35 Hz (0.33) | 19/20, mean 0.096 | 19/20, mean 0.097 | 8/20, mean 0.012 |

### Nothing coupled

Conditional M is close to the 1-in-20 rate a test at p ≤ 0.05 should have. Most directions are 0/20. The worst conditional M cell is 1/20. Pairwise M is less tidy: breathing → attention at 0.35 Hz is 4/20, and several other cells are 2/20 or 3/20. Partial cross mapping reaches 3/20 on some directions at 0.12 and 0.35 Hz. Prediction gain is mostly 0/20 to 2/20. On a genuine null, conditional M is the best calibrated of the three. Pairwise M at 0.35 Hz is the most liberal.

### Sweep of one coefficient at a time

Respiratory frequency 0.25 Hz, 120 seconds, seeds 0–19, coefficients 0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90. The other five coefficients stay zero. "Starts detecting" means the rate first reaches 80%.

False-positive rate on tests whose coefficient was zero:

| Method | All absent tests | Every coefficient zero | One other link is on, this direction is not |
| --- | --- | --- | --- |
| Pairwise Andrzejak M | 0.200 (865/4320) | 0.050 (36/720) | 0.230 (829/3600) |
| Conditional Andrzejak M | 0.145 (626/4320) | 0.017 (12/720) | 0.171 (614/3600) |
| Partial cross mapping direct-link index | 0.181 (782/4320) | 0.042 (30/720) | 0.209 (752/3600) |

When every coefficient is zero, pairwise M sits on the nominal 0.05 and conditional M is a bit stricter (0.017). The higher overall rates come from calling an absent direction present while some other link is on. That leakage is largest for pairwise M (0.230) and partial cross mapping (0.209). Conditional M still leaks, at 0.171, which is well above 0.05.

An isolated link is not reliably detected at the reference value 0.33. At coefficient 0.30, pairwise detection rates are 0.50 to 0.75 depending on direction. Every direction reaches 80% by 0.45 for pairwise M and for partial cross mapping. Conditional M reaches 80% by 0.45 on five of the six directions. Pupil diameter → attention (locus coeruleus → attention) is 15/20 at 0.45 and first reaches 80% at 0.60 (20/20). Inside the published network those same links are detected anyway, because the other couplings are on and reinforce them. A lone link at 0.33 is a harder detection than the same coefficient inside the published model.

## Caveats

* A few dozen respiratory cycles are a practical minimum. Below that, the neighbour search is noisy and the smallest p-value a short surrogate set can return is coarse. The oscillator checks use about 50 cycles. A desk test of a few seconds will run, and its p-values should not be interpreted.
* Conditional M and the forecast gain answer different questions (contemporaneous neighbourhoods, versus a short forecast). Across 20 seeds the forecast detects the direct breathing–attention link on only 8/20 seeds at 0.25 Hz and 0.35 Hz, against 17/20 to 19/20 for conditional M. A missed forecast is not evidence against a link conditional M detects. At 0.12 Hz with coefficient 0.45 the forecast does better (16/20 and 18/20).
* Perfect phase lock makes breathing and attention functions of pupil diameter even when the direct coefficient is zero. Conditional methods need residual jitter. The phase noise of 0.55 is a modelling choice, not a fitted physiological parameter. It is high enough that conditional M separates the indirect route (0/20) from the direct route (17/20 to 20/20). It is not high enough for partial cross mapping to drop the indirect route.
* Partial cross mapping, implemented from Leng et al., still calls the indirect breathing–attention route present on 15/20 to 20/20 seeds in the published model. That is a real miss for the question this analysis was built to answer. Conditional M answers it; partial cross mapping, at this noise and these frequencies, does not.
* Once every link is on, conditional M becomes redundant and misses true links. Attention → pupil given breathing falls to 8/20–10/20 in the direct model, with a mean index near zero. A non-detection there does not mean the link was absent.
* An isolated coefficient of 0.33, with the other five at zero, is only partly detected. Reliable (80%) detection of a lone link needs about 0.45, and 0.60 for conditional M on pupil → attention. Inside the published network the same 0.33 is detected because the other links reinforce it.
* Embedding delay, dimension, neighbour count, pool size, and the Theiler window move the rates. The figures above use dimension 3, k = 8, a pool of 40, and a Theiler window of one respiratory cycle. A much shorter record will be noisier: with 19 surrogates the smallest p-value is 0.05, so one seed in twenty is expected to light up even when nothing is coupled, and pairwise M at 0.35 Hz did worse than that (4/20) on one null direction.
* Multiple-comparison correction across every band, ratio, and direction is the right setting for an exploratory recording and will make weak links harder to call. The oscillator checks are planned comparisons and are not corrected.
* The hub's own `--simulate` session is a set of unrelated sine waves plus the file format. It is not the oscillator model. It checks that the command line can read a real session. The ground truth is the oscillator.

## Command

```bash
python -m sniff_sync.nlid recordings/P01
```

Useful switches: `--sample-rate`, `--group frontal`, `--channels AF3,AF4,F3,F4`, `--signals "delta/alpha power ratio"`, `--pupil`, `--pupil-offset`, `--surrogates`, `--surrogate time-shift|phase-random|iaaft`, `--correction none|bonferroni|fdr`, `--neighbors`, `--theiler`, `--max-blink`.

The table is `nlid/interdependence.csv`. The figure is `nlid/summary.png`: pairwise Andrzejak M, conditional Andrzejak M, and the partial-cross-mapping direct-link index, one row per attentional signal and one column per direction. A star marks a detection. `nlid/parameters.json` records the sample rate, the embedding delay and dimension, and any bands that were skipped.

## References

* Arnhold J, Grassberger P, Lehnertz K, Elger CE (1999). A robust method for detecting interdependences: application to intracranially recorded EEG. Physica D 134:419–430.
* Quian Quiroga R, Kraskov A, Kreuz T, Grassberger P (2002). Performance of different synchronization measures in real data: a case study on electroencephalographic signals. Physical Review E 65:041903. doi 10.1103/PhysRevE.65.041903
* Andrzejak RG, Kraskov A, Stögbauer H, Mormann F, Lehnertz K (2003). Bivariate surrogate techniques: necessity, strengths, and caveats. Physical Review E 68:066202. doi 10.1103/PhysRevE.68.066202
* Fraser AM, Swinney HL (1986). Independent coordinates for strange attractors from mutual information. Physical Review A 33:1134–1140.
* Kennel MB, Brown R, Abarbanel HDI (1992). Determining embedding dimension for phase-space reconstruction using a geometrical construction. Physical Review A 45:3403–3411.
* Theiler J (1986). Spurious dimension from correlation algorithms applied to limited time-series data. Physical Review A 34:2427–2432.
* Schreiber T, Schmitz A (1996). Improved surrogate data for nonlinearity tests. Physical Review Letters 77:635–638.
* Schreiber T, Schmitz A (2000). Surrogate time series. Physica D 142:346–382.
* Sugihara G, May R, Ye H, et al. (2012). Detecting causality in complex ecosystems. Science 338:496–500.
* Leng S, Ma H, Kurths J, et al. (2020). Partial cross mapping eliminates indirect causal influences. Nature Communications 11:2632. doi 10.1038/s41467-020-16238-0
* Melnychuk MC et al. (2018). Coupling of respiration and attention via the locus coeruleus: effects of meditation and pranayama. Psychophysiology 55:e13091. doi 10.1111/psyp.13091
* Melnychuk MC et al. (2021). Brain Sciences 11:1324. doi 10.3390/brainsci11101324
* Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society B 57:289–300.
