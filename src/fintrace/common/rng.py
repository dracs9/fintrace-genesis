"""Episode-scoped named PCG64 streams as specified in §6."""

from dataclasses import dataclass
from hashlib import sha256
from zlib import crc32

from numpy.random import Generator, PCG64, SeedSequence


@dataclass(frozen=True)
class Streams:
    seed: int
    config_hash: str
    version: str
    salt: str

    def stream(self, name: str) -> Generator:
        """Return a fresh named stream, independent of request order and other draws."""
        entropy = [self.seed, *(crc32(s.encode("utf-8")) for s in (self.config_hash, self.version, self.salt))]
        key = int.from_bytes(sha256(name.encode("utf-8")).digest()[:4], "big")
        return Generator(PCG64(SeedSequence(entropy, spawn_key=(key,))))
