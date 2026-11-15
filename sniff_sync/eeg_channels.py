"""Channel names emokit reads from an EPOC headset.

Order matches the fourteen scalp electrodes the library exposes. Gyro
axes are not EEG and are left out of the recording.
"""

EPOC_CHANNELS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]

# emokit's own FAQ: one least-significant bit of the 14-bit value is
# 0.51 microvolts on the original EPOC. Confirm this on the headset in
# front of you before treating the numbers as microvolts.
LSB_MICROVOLTS = 0.51
