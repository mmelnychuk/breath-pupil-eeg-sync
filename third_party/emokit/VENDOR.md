# Vendored emokit

Source: https://github.com/openyou/emokit
Commit: `7f25321a1c3a240f5b64a1572e5f106807b1beea`
License: public domain (see `LICENSE`). Sample recordings that shipped in the repository were not copied.

emokit reads the raw electroencephalography (EEG) stream from an Emotiv headset over USB (Universal Serial Bus) human-interface-device (HID) access. It does not use Emotiv's Cortex service.

## Headsets

Supported by upstream: EPOC, and EPOC+ from before 2016 (gyro axes are limited).

Not supported: EPOC+ from 2016 onward, EPOC X, Insight, Flex, MN8, and any headset that only speaks Cortex.

## Python

The source contains branches for Python 3. Importing it on a current Python still fails until two old packages are installed: `pycrypto==2.6.1` (the Advanced Encryption Standard (AES) decryptor) and `hidapi`. `pycrypto` 2.6.1 does not build on Python 3.12. This project does not pretend otherwise. A real headset has to be tried on the machine's own Python.

## Decryption for the original EPOC

The decryption code was not changed. An original consumer EPOC uses `crypto_key` in `util.py`: a 16-byte Advanced Encryption Standard (AES) key built from the headset serial number, then AES in ECB mode. The EPOC+ key (`new_crypto_key` / `epoc_plus_crypto_key`) is a different recipe. This project opens the headset with `is_research=False` and `force_old_crypto=True`, so the reader takes the original `crypto_key` path and the old 14-bit packet layout.

## Fixes applied in this copy

Neither fix touches the key or the decrypt function.

- `python/emokit/reader.py`: the reader thread printed `ex.message`, which exists on Python 2 only. A read error ended the thread. It now prints the exception itself.
- `python/emokit/emotiv.py`: `serial_number.startswith` ran even when no headset had been found and the serial was `None`.

## Clock

emokit stamps a packet with `datetime.now()` (local wall time, as a datetime). This project does not use that value. When a packet is dequeued it is stamped with `time.time()` on this computer, and that Unix time is what is saved.
