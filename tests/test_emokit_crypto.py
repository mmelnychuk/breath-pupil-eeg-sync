"""Decryption of an original consumer EPOC packet with pycryptodome."""

import sys
import unittest
from pathlib import Path

_EMOKIT = Path(__file__).resolve().parents[1] / "third_party" / "emokit" / "python"
if str(_EMOKIT) not in sys.path:
    sys.path.insert(0, str(_EMOKIT))

from sniff_sync.eeg import emokit_status
from sniff_sync.eeg_channels import EPOC_CHANNELS


class EmokitCryptoTests(unittest.TestCase):
    def test_emokit_imports_with_pycryptodome_on_this_python(self):
        self.assertGreaterEqual(sys.version_info[:2], (3, 9))
        status = emokit_status()
        self.assertEqual(status["import_error"], "")
        self.assertIn("original EPOC", status["supported_models"])
        import Crypto
        from Crypto.Cipher import AES

        self.assertTrue(AES)
        self.assertTrue(Crypto.__version__)

    def test_consumer_key_schedule_is_unchanged(self):
        from emokit.util import crypto_key

        serial = "SN123456"
        # Consumer headset, not the research edition. Characters are the last
        # four of the serial number, then the fixed original-EPOC bytes.
        expected = bytearray(16)
        expected[0] = ord(serial[-1])
        expected[1] = 0
        expected[2] = ord(serial[-2])
        expected[3] = ord("T")
        expected[4] = ord(serial[-3])
        expected[5] = 0x10
        expected[6] = ord(serial[-4])
        expected[7] = ord("B")
        expected[8] = ord(serial[-1])
        expected[9] = 0
        expected[10] = ord(serial[-2])
        expected[11] = ord("H")
        expected[12] = ord(serial[-3])
        expected[13] = 0
        expected[14] = ord(serial[-4])
        expected[15] = ord("P")
        self.assertEqual(crypto_key(serial, is_research=False).encode("latin-1"), bytes(expected))

    def test_encrypted_packet_round_trips_to_the_fourteen_channels(self):
        from emokit.decrypter import EmotivCrypto, decrypt_data
        from emokit.packet import EmotivOldPacket
        from emokit.sensors import sensors_14_bits

        serial = "SN123456"
        plaintext = bytearray(32)
        plaintext[0] = 7
        # Distinct 14-bit levels. Overlapping bit positions are resolved by
        # writing the channels in electrode order; the expected value is read
        # back from the finished plaintext with the same bit positions.
        levels = {
            "AF3": 1000,
            "F7": 1100,
            "F3": 1200,
            "FC5": 1300,
            "T7": 1400,
            "P7": 1500,
            "O1": 1600,
            "O2": 1700,
            "P8": 1800,
            "T8": 1900,
            "FC6": 2000,
            "F4": 2100,
            "F8": 2200,
            "AF4": 2300,
        }
        for name in EPOC_CHANNELS:
            _write_level(plaintext, sensors_14_bits[name], levels[name])
        expected = {
            name: _read_level(plaintext, sensors_14_bits[name]) * 0.5151515151 for name in EPOC_CHANNELS
        }
        crypto = EmotivCrypto(serial_number=serial, is_research=False, force_old_crypto=True)
        cipher = crypto.new_cipher()
        raw = bytes(plaintext)
        encrypted = cipher.encrypt(raw[:16]) + cipher.encrypt(raw[16:])
        decrypted = decrypt_data(cipher, encrypted)
        self.assertEqual(decrypted, raw)
        packet = EmotivOldPacket(decrypted)
        self.assertEqual(len(EPOC_CHANNELS), 14)
        for name in EPOC_CHANNELS:
            self.assertEqual(packet.sensors[name]["value"], expected[name])


def _write_level(data, bits, value):
    for index in range(14):
        bit_index = bits[index]
        byte_index = (bit_index // 8) + 1
        offset = bit_index % 8
        if (int(value) >> index) & 1:
            data[byte_index] |= 1 << offset
        else:
            data[byte_index] &= ~(1 << offset) & 0xFF


def _read_level(data, bits):
    level = 0
    for index in range(13, -1, -1):
        level <<= 1
        bit_index = bits[index]
        byte_index = (bit_index // 8) + 1
        offset = bit_index % 8
        level |= (data[byte_index] >> offset) & 1
    return level


if __name__ == "__main__":
    unittest.main()
