"""Read single members of a large remote zip with HTTP range requests (no full download).

Usage:
  python scripts/remote_zip.py list URL
  python scripts/remote_zip.py get  URL MEMBER OUT_PATH
"""
import io
import time
import sys
import urllib.error
import urllib.request
import zipfile


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP; each read is one Range request."""

    def __init__(self, url: str):
        self.orig, self.pos = url, 0
        self._resolve()

    def _resolve(self):
        """Follow the CDN redirect. The redirect target is a signed URL that expires (403 after
        about an hour), so it is resolved again whenever a range request is refused."""
        req = urllib.request.Request(self.orig, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url = r.geturl()
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: off, io.SEEK_CUR: self.pos + off, io.SEEK_END: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        for attempt in range(8):
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                if len(data) != end - self.pos + 1:
                    raise IOError(f"short read: {len(data)} of {end - self.pos + 1} bytes")
                break
            except Exception as e:
                if attempt == 7:
                    raise
                if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 410):
                    self._resolve()  # signed URL expired: get a fresh one
                time.sleep(min(30, 2 ** attempt))
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    cmd, url = sys.argv[1], sys.argv[2]
    z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=8 << 20))
    if cmd == "list":
        for i in z.infolist():
            print(f"{i.file_size / 1e6:10.1f} MB  {i.compress_size / 1e6:10.1f} MB packed  {i.filename}")
    elif cmd == "get":
        member, out = sys.argv[3], sys.argv[4]
        with z.open(member) as src, open(out, "wb") as dst:
            while chunk := src.read(8 << 20):
                dst.write(chunk)
        print("saved", out)


if __name__ == "__main__":
    main()
