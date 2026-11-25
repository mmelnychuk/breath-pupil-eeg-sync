# Vendored emokit

Source: https://github.com/openyou/emokit
Commit: `7f25321a1c3a240f5b64a1572e5f106807b1beea`
License: public domain (see `LICENSE`). Sample recordings that shipped in the repository were not copied.

emokit reads the raw electroencephalography (EEG) stream from an Emotiv headset over USB (Universal Serial Bus) human-interface-device (HID) access. It does not use Emotiv's Cortex service.

## Headsets

Supported by upstream: EPOC, and EPOC+ from before 2016 (gyro axes are limited).

Not supported: EPOC+ from 2016 onward, EPOC X, Insight, Flex, MN8, and any headset that only speaks Cortex.

## Python

The source contains branches for Python 3. This copy decrypts with `pycryptodome` (import name `Crypto.Cipher.AES`) instead of `pycrypto` 2.6.1, which does not build on Python 3.12. It was checked on Python 3.12. The human-interface-device library is still installed separately as `hidapi`. Current releases of that package are imported as `hid`, so `python/emokit/hidapi_compat.py` supplies the function names the reader already calls.

## Decryption for the original EPOC

The key schedule was not changed. An original consumer EPOC uses `crypto_key` in `util.py`: a 16-byte Advanced Encryption Standard key built from the headset serial number, then electronic codebook mode. The EPOC+ key (`new_crypto_key` / `epoc_plus_crypto_key`) is a different recipe and is not used. This project opens the headset with `is_research=False` and `force_old_crypto=True`, so the reader takes the original `crypto_key` path and the old 14-bit packet layout.

`pycryptodome` requires the key as bytes and rejects an initialisation vector in electronic codebook mode. The older library accepted a vector and ignored it. The cipher is now created without that vector. The key bytes are the same string the schedule already returned, encoded as Latin-1 so the zero bytes are preserved.

## Fixes applied in this copy

None of these edits change the key schedule or which bits belong to which electrode.

- `python/emokit/reader.py`: the reader thread printed `ex.message`, which exists on Python 2 only. A read error ended the thread. It now prints the exception itself. The human-interface-device library is imported when a headset is opened, not when the module is imported.
- `python/emokit/emotiv.py`: `serial_number.startswith` ran even when no headset had been found and the serial was `None`.
- `python/emokit/decrypter.py`: the cipher is built with `pycryptodome`, the key is passed as bytes, and electronic codebook mode is given no initialisation vector.
- `python/emokit/util.py`: `get_level` called `ord` on every packet byte. On Python 3 those bytes are already integers, so parsing a decrypted packet raised `TypeError` before any channel value was produced. Integers are now used directly. The bit positions are unchanged.
- `python/emokit/output.py`: `from packet import ...` could not see the package on Python 3. It is now a relative import.
- `python/emokit/hidapi_compat.py`: adapts `import hid` to the older `hidapi.hid_*` names.

## Clock

emokit stamps a packet with `datetime.now()` (local wall time, as a datetime). This project does not use that value. When a packet is dequeued it is stamped with `time.time()` on this computer, and that Unix time is what is saved.
