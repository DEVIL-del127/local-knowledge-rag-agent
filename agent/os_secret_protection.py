"""Current-Windows-user DPAPI protection, without files or plaintext fallback."""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


class SecretProtectionUnavailable(RuntimeError):
    pass


class _Blob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


class WindowsUserProtection:
    """Protect bytes for the current Windows user (not machine-wide).

    WSL must use a separately verified bridge; this class does not run shells,
    move secrets through environment variables, or silently select another user.
    """

    def __init__(self):
        if os.name != "nt":
            raise SecretProtectionUnavailable("windows_user_protection_unavailable")
        self._crypt = ctypes.WinDLL("crypt32", use_last_error=True)
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel.LocalFree.argtypes = [ctypes.c_void_p]
        self._kernel.LocalFree.restype = ctypes.c_void_p

    def protect(self, plaintext: bytes, *, context: bytes) -> bytes:
        return self._transform(plaintext, context=context, decrypt=False)

    def unprotect(self, ciphertext: bytes, *, context: bytes) -> bytes:
        return self._transform(ciphertext, context=context, decrypt=True)

    def _transform(self, payload: bytes, *, context: bytes, decrypt: bool) -> bytes:
        if not isinstance(payload, bytes) or not payload or len(payload) > 16 * 1024 * 1024:
            raise ValueError("invalid protected payload size")
        if not isinstance(context, bytes) or not context or len(context) > 4096:
            raise ValueError("invalid protection context")

        data_buffer = ctypes.create_string_buffer(payload)
        context_buffer = ctypes.create_string_buffer(context)
        source = _Blob(len(payload), ctypes.cast(data_buffer, ctypes.POINTER(ctypes.c_ubyte)))
        entropy = _Blob(len(context), ctypes.cast(context_buffer, ctypes.POINTER(ctypes.c_ubyte)))
        output = _Blob()
        fn = self._crypt.CryptUnprotectData if decrypt else self._crypt.CryptProtectData
        fn.argtypes = [ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.POINTER(_Blob),
                       ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)]
        fn.restype = wintypes.BOOL
        try:
            # CRYPTPROTECT_UI_FORBIDDEN; deliberately omit LOCAL_MACHINE.
            if not fn(ctypes.byref(source), None, ctypes.byref(entropy), None, None, 1, ctypes.byref(output)):
                raise SecretProtectionUnavailable("windows_user_protection_failed")
            return ctypes.string_at(output.data, output.size)
        finally:
            if output.data:
                ctypes.memset(output.data, 0, output.size)
                self._kernel.LocalFree(ctypes.cast(output.data, ctypes.c_void_p))
            ctypes.memset(data_buffer, 0, ctypes.sizeof(data_buffer))
            ctypes.memset(context_buffer, 0, ctypes.sizeof(context_buffer))
