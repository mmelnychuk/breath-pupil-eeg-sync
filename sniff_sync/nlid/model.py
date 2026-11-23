"""Stochastic Kuramoto simulator for the respiration–locus coeruleus–attention model.

Melnychuk et al. (2018), Psychophysiology, doi 10.1111/psyp.13091, equation 2,
write a respiratory oscillator, a locus-coeruleus oscillator, and an attentional
oscillator. The printed paper puts the locus-coeruleus coefficient on the
respiratory line. The author confirmed that this is a typographical error:
respiration has its own coefficient. There is no direct breathing–attention
term in that equation. The 2021 follow-up found linear Granger causality in
every direction, including a direct breathing–cortex link, so the simulator
can add that term.

Each directed link has its own coefficient. The coupling function is sinusoidal,
as in the Kuramoto model:

    dφ_A / dt  = ω_A  + ε_{LC→A} sin(φ_LC − φ_A) + ε_{R→A} sin(φ_R − φ_A) + ξ_A
    dφ_LC / dt = ω_LC + ε_{A→LC} sin(φ_A − φ_LC) + ε_{R→LC} sin(φ_R − φ_LC) + ξ_LC
    dφ_R / dt  = ω_R  + ε_{LC→R} sin(φ_LC − φ_R) + ε_{A→R} sin(φ_A − φ_R) + ξ_R

ξ is Gaussian white noise. The state is integrated with the Euler–Maruyama method.

Figure 5 of the 2018 paper used one coefficient, 0.33, for a stable limit cycle
at respiratory frequencies 0.35 Hz and 0.25 Hz, and 0.45 at 0.12 Hz. The author
describes these as values found by searching the parameter space, not as
physiological constants. ``reference_epsilon`` returns those values. The
published scenario puts  that coefficient on the four model links (locus
coeruleus ↔ attention and locus coeruleus ↔ respiration) and leaves the two
breathing–attention links at zero. The direct-link scenario sets those two
links to the same coefficient.

Phases are reported through one observable each, plus a little measurement noise:

    breathing       = sin(φ_R)
    pupil diameter  = sin(φ_LC) + 0.3 sin(2 φ_LC)
    attention       = sin(φ_A)

Pupil diameter here is an arbitrary unit standing in for the locus-coeruleus
phase, not a millimetre measurement. The second harmonic makes the pupil
waveform asymmetric, closer to hippus than a pure sine, without hiding the phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Order used everywhere in this module: attention, locus coeruleus, respiration.
ATTENTION = 0
LOCUS = 1
RESPIRATION = 2

LINK_NAMES = (
    "locus_to_attention",
    "attention_to_locus",
    "respiration_to_locus",
    "locus_to_respiration",
    "respiration_to_attention",
    "attention_to_respiration",
)

# (source index, target index) in the phase vector.
LINK_INDEX = {
    "locus_to_attention": (LOCUS, ATTENTION),
    "attention_to_locus": (ATTENTION, LOCUS),
    "respiration_to_locus": (RESPIRATION, LOCUS),
    "locus_to_respiration": (LOCUS, RESPIRATION),
    "respiration_to_attention": (RESPIRATION, ATTENTION),
    "attention_to_respiration": (ATTENTION, RESPIRATION),
}

OBSERVABLE_OF_LINK = {
    "locus_to_attention": ("pupil diameter", "attention"),
    "attention_to_locus": ("attention", "pupil diameter"),
    "respiration_to_locus": ("breathing", "pupil diameter"),
    "locus_to_respiration": ("pupil diameter", "breathing"),
    "respiration_to_attention": ("breathing", "attention"),
    "attention_to_respiration": ("attention", "breathing"),
}


@dataclass(frozen=True)
class Coupling:
    """One coefficient per directed link. Units are radians per second."""

    locus_to_attention: float = 0.0
    attention_to_locus: float = 0.0
    respiration_to_locus: float = 0.0
    locus_to_respiration: float = 0.0
    respiration_to_attention: float = 0.0
    attention_to_respiration: float = 0.0

    def matrix(self):
        """Square matrix indexed [target, source]."""
        out = np.zeros((3, 3))
        for name in LINK_NAMES:
            source, target = LINK_INDEX[name]
            out[target, source] = float(getattr(self, name))
        return out

    def as_dict(self):
        return {name: float(getattr(self, name)) for name in LINK_NAMES}


def reference_epsilon(respiratory_hz):
    """Coefficient used for Figure 5 at this respiratory frequency.

    0.45 at 0.12 Hz, and 0.33 at 0.25 Hz and 0.35 Hz. Any other frequency
    below 0.16 Hz uses 0.45; faster frequencies use 0.33.
    """
    if float(respiratory_hz) <= 0.16:
        return 0.45
    return 0.33


def reference_coupling(respiratory_hz):
    """Published model: four links at the reference coefficient, no direct breathing–attention link."""
    epsilon = reference_epsilon(respiratory_hz)
    return Coupling(
        locus_to_attention=epsilon,
        attention_to_locus=epsilon,
        respiration_to_locus=epsilon,
        locus_to_respiration=epsilon,
        respiration_to_attention=0.0,
        attention_to_respiration=0.0,
    )


def direct_coupling(respiratory_hz):
    """Published model plus a direct breathing–attention link in both directions."""
    epsilon = reference_epsilon(respiratory_hz)
    base = reference_coupling(respiratory_hz)
    return Coupling(
        locus_to_attention=base.locus_to_attention,
        attention_to_locus=base.attention_to_locus,
        respiration_to_locus=base.respiration_to_locus,
        locus_to_respiration=base.locus_to_respiration,
        respiration_to_attention=epsilon,
        attention_to_respiration=epsilon,
    )


def uncoupled():
    """Every directed coefficient is zero."""
    return Coupling()


def phase_drift(phase, angular_frequency, coupling_matrix):
    """Right-hand side of the phase equations, without the noise."""
    phase = np.asarray(phase, dtype=float)
    # difference[target, source] = phase[source] − phase[target]
    difference = phase[None, :] - phase[:, None]
    drive = (np.asarray(coupling_matrix, dtype=float) * np.sin(difference)).sum(axis=1)
    return np.asarray(angular_frequency, dtype=float) + drive


def simulate_oscillators(
    coupling,
    respiratory_hz=0.25,
    duration_s=200.0,
    sample_rate_hz=10.0,
    step_s=0.005,
    phase_noise=0.55,
    observation_noise=0.05,
    locus_frequency_offset_hz=-0.03,
    attention_frequency_offset_hz=0.04,
    seed=0,
):
    """Integrate the three phase equations and return the observable signals.

    Natural frequencies sit near the respiratory frequency. The locus coeruleus
    is ``locus_frequency_offset_hz`` away from it and attention is
    ``attention_frequency_offset_hz`` away. Offsets of a few hundredths of a
    hertz still phase-lock at the reference coefficients, and they keep three
    uncoupled oscillators from sharing one period by accident.

    ``phase_noise`` is the intensity of the Wiener process, in radians per
    square-root second. The noise-free equations produce a single limit cycle
    on which every phase is a function of every other phase, so a direct link
    cannot be told from a link that only passes through the locus coeruleus.
    Moderate phase noise keeps the coupling and leaves enough independent
    jitter for that distinction. It is not a value from the 2018 paper.
    """
    generator = np.random.default_rng(seed)
    respiratory_hz = float(respiratory_hz)
    if respiratory_hz <= 0:
        raise ValueError("The respiratory frequency must be positive.")
    locus_hz = respiratory_hz + float(locus_frequency_offset_hz)
    attention_hz = respiratory_hz + float(attention_frequency_offset_hz)
    if min(locus_hz, attention_hz, respiratory_hz) <= 0:
        raise ValueError("Frequency offsets push a natural frequency to or below zero.")
    angular = 2.0 * np.pi * np.array([attention_hz, locus_hz, respiratory_hz])
    matrix = coupling.matrix() if isinstance(coupling, Coupling) else Coupling(**coupling).matrix()
    n_steps = int(round(float(duration_s) / float(step_s)))
    sample_every = int(round((1.0 / float(sample_rate_hz)) / float(step_s)))
    if sample_every < 1 or n_steps < sample_every * 10:
        raise ValueError("The integration step and the sample rate do not leave enough samples.")
    n_out = n_steps // sample_every
    phase = generator.uniform(0.0, 2.0 * np.pi, size=3)
    recorded = np.empty((n_out, 3))
    noise_scale = float(phase_noise) * np.sqrt(float(step_s))
    cursor = 0
    for step in range(n_steps):
        if step % sample_every == 0 and cursor < n_out:
            recorded[cursor] = phase
            cursor += 1
        drift = phase_drift(phase, angular, matrix)
        phase = phase + drift * float(step_s) + noise_scale * generator.normal(size=3)
    phase_attention = recorded[:, ATTENTION]
    phase_locus = recorded[:, LOCUS]
    phase_respiration = recorded[:, RESPIRATION]
    breathing = np.sin(phase_respiration)
    pupil = np.sin(phase_locus) + 0.3 * np.sin(2.0 * phase_locus)
    attention = np.sin(phase_attention)
    if observation_noise:
        breathing = breathing + float(observation_noise) * generator.normal(size=n_out)
        pupil = pupil + float(observation_noise) * generator.normal(size=n_out)
        attention = attention + float(observation_noise) * generator.normal(size=n_out)
    time = np.arange(n_out) / float(sample_rate_hz)
    return {
        "time_s": time,
        "breathing": breathing,
        "pupil diameter": pupil,
        "attention": attention,
        "phase_attention": phase_attention,
        "phase_locus": phase_locus,
        "phase_respiration": phase_respiration,
        "respiratory_hz": respiratory_hz,
        "locus_hz": locus_hz,
        "attention_hz": attention_hz,
        "sample_rate_hz": float(sample_rate_hz),
        "coupling": matrix,
    }


def scenario_coupling(name, respiratory_hz):
    """``published``, ``direct``, or ``uncoupled``."""
    key = name.strip().lower()
    if key == "published":
        return reference_coupling(respiratory_hz)
    if key == "direct":
        return direct_coupling(respiratory_hz)
    if key in ("uncoupled", "null", "zero"):
        return uncoupled()
    raise ValueError("Unknown scenario %r. Use published, direct, or uncoupled." % name)
