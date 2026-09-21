import pytest

from agent.privacy import PersistenceCipher, PersistencePolicyError


def test_plaintext_is_not_accepted_as_encrypted_state(tmp_path):
    cipher = PersistenceCipher(tmp_path / "key")
    with pytest.raises(PersistencePolicyError):
        cipher.decrypt_json('{"private":"synthetic"}', aad="state")
    value = {"private": "synthetic"}
    envelope = cipher.encrypt_json(value, aad="state")
    assert cipher.decrypt_json(envelope, aad="state") == value
    with pytest.raises(Exception):
        cipher.decrypt_json(envelope, aad="other-state")
