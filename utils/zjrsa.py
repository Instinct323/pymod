from pathlib import Path
from typing import Union

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

pad = padding.OAEP(
    mgf=padding.MGF1(algorithm=hashes.SHA256()),
    algorithm=hashes.SHA256(),
    label=None
)


def file2bytes(src: Union[bytes, Path]) -> bytes:
    return src.read_bytes() if isinstance(src, Path) else src


def bytes2str(src: bytes) -> str:
    return ''.join(map(chr, src))


def str2bytes(src: str) -> bytes:
    return bytes(map(ord, src))


class PublicKey:

    def __init__(self,
                 pem: Union[bytes, Path] = None,
                 key: rsa.RSAPublicKey = None):
        self.key = serialization.load_pem_public_key(file2bytes(pem)) if pem else key
        assert self.key, 'Invalid initialization parameter'

    def __call__(self,
                 src: Union[bytes, Path],
                 dst: Path = None):
        ciphertext = self.key.encrypt(file2bytes(src), padding=pad)
        if dst: dst.write_bytes(ciphertext)
        return ciphertext

    def dump_pem(self,
                 dst: Path = None):
        pem = self.key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        if dst: dst.write_bytes(pem)
        return pem


class PrivateKey:

    def __init__(self,
                 pem: Union[bytes, Path] = None,
                 key_size: int = 2048):
        self.key = serialization.load_pem_private_key(file2bytes(pem)) if pem \
            else rsa.generate_private_key(public_exponent=65537, key_size=key_size)

    def __call__(self,
                 src: Union[bytes, Path],
                 dst: Path = None):
        plaintext = self.key.decrypt(file2bytes(src), padding=pad)
        if dst: dst.write_bytes(plaintext)
        return plaintext

    def public_key(self) -> PublicKey:
        return PublicKey(key=self.key.public_key())

    def dump_pem(self,
                 dst: Path = None):
        pem = self.key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )
        if dst: dst.write_bytes(pem)
        return pem


if __name__ == '__main__':
    import time
    import os

    os.chdir(r'D:\Information\Download')
    pub_f = Path('public.pem')
    bin_f = Path('msg.bin')

    if 1:
        # 优先运行
        pri = PrivateKey()
        pub = pri.public_key().dump_pem(pub_f)
        print(pub.decode('utf-8'))

        input('wait: ')
        t0 = time.time()
        print(pri(bin_f))
        print(time.time() - t0)

    else:
        pub = PublicKey(pem=pub_f)
        msg = b'aa'
        cipher = pub(msg, bin_f)
        print(bytes2str(cipher))
