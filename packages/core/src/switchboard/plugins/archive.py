"""Leitura segura do pacote de um plugin (diretório, ``.zip`` ou bytes) e empacotamento.

O pacote é lido inteiro para a memória, com limites, e nada é gravado em disco nem
executado. Recusa o que um arquivo malicioso costuma usar: caminhos que escapam da
raiz (zip-slip), links simbólicos, entradas criptografadas ou duplicadas, bombas de
compressão e qualquer tipo de arquivo que não seja texto declarativo.
"""

from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .findings import Finding

MiB = 1024 * 1024

# só texto declarativo: o router nunca executa nada de um plugin
ALLOWED_EXTENSIONS = frozenset(
    {".yaml", ".yml", ".json", ".md", ".markdown", ".txt", ".csv", ".rst", ".html", ".htm"}
)
ALLOWED_BARE_NAMES = frozenset({"LICENSE", "NOTICE", "README", "CHANGELOG", "AUTHORS"})
# lixo de sistema e de controle de versão: ignorado em silêncio
IGNORED_NAMES = frozenset(
    {".DS_Store", "Thumbs.db", ".gitignore", ".gitattributes", ".editorconfig", "desktop.ini"}
)
IGNORED_DIRS = frozenset({".git", "__MACOSX", "__pycache__", ".idea", ".vscode"})

MAX_PATH_CHARS = 200


@dataclass(frozen=True)
class Limits:
    max_archive_bytes: int = 10 * MiB
    max_total_bytes: int = 25 * MiB
    max_file_bytes: int = 2 * MiB
    max_files: int = 300
    max_ratio: int = 200  # tamanho descompactado / compactado, acima disso é bomba de compressão


DEFAULT_LIMITS = Limits()


def digest_files(files: dict[str, bytes]) -> str:
    """SHA-256 do conteúdo (caminhos e bytes), independente de como foi empacotado."""
    h = hashlib.sha256()
    for path in sorted(files):
        h.update(path.encode("utf-8") + b"\0")
        h.update(hashlib.sha256(files[path]).digest())
    return h.hexdigest()


@dataclass
class Package:
    files: dict[str, bytes] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    @property
    def digest(self) -> str:
        return digest_files(self.files)


def _block(code: str, message: str, where: str = "") -> Finding:
    return Finding("block", code, message, where)


def clean_path(raw: str) -> tuple[str | None, str | None]:
    """Normaliza o caminho de uma entrada; devolve ``(caminho, problema)``."""
    if not raw or "\x00" in raw:
        return None, "nome vazio ou com byte nulo"
    if "\\" in raw:
        return None, "caminho com '\\' (use '/')"
    if raw.startswith("/") or (len(raw) > 1 and raw[1] == ":"):
        return None, "caminho absoluto"
    if len(raw) > MAX_PATH_CHARS:
        return None, f"caminho com mais de {MAX_PATH_CHARS} caracteres"
    parts = PurePosixPath(raw).parts
    if any(p in ("..", "") for p in parts):
        return None, "caminho que sai da raiz do pacote ('..')"
    if any(ord(ch) < 32 or ch == "\x7f" for ch in raw):
        return None, "caminho com caractere de controle"
    return "/".join(p for p in parts if p != "."), None


def _skipped(path: str) -> bool:
    parts = path.split("/")
    return parts[-1] in IGNORED_NAMES or any(p in IGNORED_DIRS for p in parts[:-1])


def _allowed_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in ALLOWED_EXTENSIONS:
        return True
    return suffix == "" and name in ALLOWED_BARE_NAMES


def _disallowed(path: str) -> Finding:
    name = path.rsplit("/", 1)[-1]
    suffix = PurePosixPath(name).suffix.lower() or "(sem extensão)"
    return _block(
        "package.disallowed_file",
        f"arquivo {suffix} não permitido: plugins só aceitam texto declarativo "
        f"({', '.join(sorted(ALLOWED_EXTENSIONS))}); o router nunca executa código de plugin",
        path,
    )


def _strip_root(files: dict[str, bytes]) -> dict[str, bytes]:
    """Pacote zipado com a pasta inclusa (``meu-plugin/plugin.yaml``): sobe um nível."""
    from .manifest import MANIFEST_FILES

    manifests = (*MANIFEST_FILES, ".claude-plugin/plugin.json")
    if any(m in files for m in manifests) or not files:
        return files
    tops = {p.split("/", 1)[0] for p in files if "/" in p}
    if len(tops) != 1 or any("/" not in p for p in files):
        return files
    prefix = next(iter(tops)) + "/"
    stripped = {p[len(prefix) :]: data for p, data in files.items()}
    return stripped if any(m in stripped for m in manifests) else files


def read_zip(data: bytes, limits: Limits = DEFAULT_LIMITS) -> Package:
    """Lê o .zip em memória. Um arquivo malformado vira o achado ``package.corrupt``, não um erro."""
    try:
        return _read_zip(data, limits)
    except Exception:  # cabeçalhos e blocos inválidos de formas que o zipfile não prevê
        pkg = Package()
        pkg.findings.append(
            _block("package.corrupt", "o arquivo está corrompido ou não pôde ser lido")
        )
        return pkg


def _read_zip(data: bytes, limits: Limits = DEFAULT_LIMITS) -> Package:
    pkg = Package()
    if len(data) > limits.max_archive_bytes:
        pkg.findings.append(
            _block(
                "package.too_large",
                f"o arquivo tem {len(data) / MiB:.1f} MB; o limite é {limits.max_archive_bytes // MiB} MB",
            )
        )
        return pkg
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        pkg.findings.append(_block("package.not_zip", "não é um arquivo .zip válido"))
        return pkg
    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > limits.max_files:
            pkg.findings.append(
                _block(
                    "package.too_many_files",
                    f"{len(infos)} arquivos; o limite é {limits.max_files}",
                )
            )
            return pkg
        total = 0
        seen: set[str] = set()
        for info in infos:
            path, problem = clean_path(info.filename)
            if problem or path is None:
                pkg.findings.append(
                    _block(
                        "package.unsafe_path", f"entrada recusada: {problem}", info.filename[:120]
                    )
                )
                continue
            if _skipped(path):
                continue
            if path in seen:
                pkg.findings.append(
                    _block("package.duplicate_entry", "o mesmo caminho aparece duas vezes", path)
                )
                continue
            seen.add(path)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                pkg.findings.append(
                    _block("package.symlink", "links simbólicos não são permitidos", path)
                )
                continue
            if info.flag_bits & 0x1:
                pkg.findings.append(
                    _block("package.encrypted", "arquivo criptografado dentro do pacote", path)
                )
                continue
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                # bzip2/LZMA descompactam blocos inteiros antes de respeitar o limite de leitura
                pkg.findings.append(
                    _block(
                        "package.compression",
                        "método de compressão não aceito (use deflate, o padrão do zip)",
                        path,
                    )
                )
                continue
            if not _allowed_file(path):
                pkg.findings.append(_disallowed(path))
                continue
            if info.file_size > limits.max_file_bytes:
                pkg.findings.append(
                    _block(
                        "package.file_too_large",
                        f"{info.file_size / MiB:.1f} MB; o limite por arquivo é "
                        f"{limits.max_file_bytes // MiB} MB",
                        path,
                    )
                )
                continue
            if (
                info.file_size > MiB // 4
                and info.file_size / max(info.compress_size, 1) > limits.max_ratio
            ):
                pkg.findings.append(
                    _block("package.zip_bomb", "taxa de compressão suspeita (bomba de zip)", path)
                )
                continue
            total += info.file_size
            if total > limits.max_total_bytes:
                pkg.findings.append(
                    _block(
                        "package.too_large",
                        f"conteúdo descompactado passa de {limits.max_total_bytes // MiB} MB",
                    )
                )
                return pkg
            try:
                with zf.open(info) as handle:
                    content = handle.read(limits.max_file_bytes + 1)
            except (zipfile.BadZipFile, RuntimeError, NotImplementedError, OSError) as exc:
                pkg.findings.append(_block("package.corrupt", f"não consegui ler: {exc}", path))
                continue
            if len(content) > limits.max_file_bytes:
                pkg.findings.append(
                    _block("package.file_too_large", "maior que o declarado no zip", path)
                )
                continue
            pkg.files[path] = content
    pkg.files = _strip_root(pkg.files)
    return pkg


def read_dir(root: Path, limits: Limits = DEFAULT_LIMITS) -> Package:
    pkg = Package()
    root = Path(root)
    if not root.is_dir():
        pkg.findings.append(_block("package.not_found", f"diretório não encontrado: {root}"))
        return pkg
    total = 0
    count = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(directory.iterdir())
        except OSError as exc:
            pkg.findings.append(_block("package.unreadable", f"não consegui listar: {exc}"))
            continue
        for entry in entries:
            rel = entry.relative_to(root).as_posix()
            if entry.is_symlink():
                if entry.name not in IGNORED_NAMES and entry.name not in IGNORED_DIRS:
                    pkg.findings.append(
                        _block("package.symlink", "links simbólicos não são permitidos", rel)
                    )
                continue
            if entry.is_dir():
                if entry.name not in IGNORED_DIRS:
                    stack.append(entry)
                continue
            if not entry.is_file() or _skipped(rel):
                continue
            path, problem = clean_path(rel)
            if problem or path is None:
                pkg.findings.append(
                    _block("package.unsafe_path", f"entrada recusada: {problem}", rel)
                )
                continue
            if not _allowed_file(path):
                pkg.findings.append(_disallowed(path))
                continue
            count += 1
            if count > limits.max_files:
                pkg.findings.append(
                    _block("package.too_many_files", f"mais de {limits.max_files} arquivos")
                )
                return pkg
            size = entry.stat().st_size
            if size > limits.max_file_bytes:
                pkg.findings.append(
                    _block(
                        "package.file_too_large",
                        f"{size / MiB:.1f} MB; o limite por arquivo é {limits.max_file_bytes // MiB} MB",
                        path,
                    )
                )
                continue
            total += size
            if total > limits.max_total_bytes:
                pkg.findings.append(
                    _block(
                        "package.too_large",
                        f"conteúdo passa de {limits.max_total_bytes // MiB} MB",
                    )
                )
                return pkg
            try:
                pkg.files[path] = entry.read_bytes()
            except OSError as exc:
                pkg.findings.append(_block("package.unreadable", f"não consegui ler: {exc}", path))
    return pkg


def read_source(
    source: str | Path | bytes, *, limits: Limits = DEFAULT_LIMITS
) -> tuple[Package, str]:
    """Lê um diretório, um ``.zip`` em disco ou os bytes de um ``.zip``; devolve também a origem."""
    if isinstance(source, bytes):
        return read_zip(source, limits), "upload"
    path = Path(source)
    if path.is_dir():
        return read_dir(path, limits), f"diretório {path.name}"
    if path.is_file():
        if path.stat().st_size > limits.max_archive_bytes:
            pkg = Package()
            pkg.findings.append(
                _block(
                    "package.too_large",
                    f"o arquivo tem mais de {limits.max_archive_bytes // MiB} MB",
                )
            )
            return pkg, f"arquivo {path.name}"
        return read_zip(path.read_bytes(), limits), f"arquivo {path.name}"
    pkg = Package()
    pkg.findings.append(_block("package.not_found", f"não encontrei {path}"))
    return pkg, str(path)


def pack(files: dict[str, bytes]) -> bytes:
    """Zip determinístico (ordem fixa, data fixa): o mesmo conteúdo dá sempre os mesmos bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(files):
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            zf.writestr(info, files[path])
    return buffer.getvalue()
