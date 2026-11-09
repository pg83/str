"""Strict byte comparison of every object in two library archives."""
import argparse
import hashlib
from pathlib import Path

from post import read_archive


def compare(reference, candidate):
    a, b = dict(read_archive(reference)), dict(read_archive(candidate))
    same, different = [], []
    for name in sorted(a.keys() | b.keys()):
        if name not in a or name not in b:
            different.append('%s: %s' % (name, 'missing from candidate' if name not in b else 'extra object'))
        elif a[name] == b[name]:
            same.append(name)
        else:
            x, y = a[name], b[name]
            offset = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            different.append('%s: first difference at %#x, %d / %d bytes; sha256 %s / %s' %
                             (name, offset, len(x), len(y), hashlib.sha256(x).hexdigest(), hashlib.sha256(y).hexdigest()))
    lines = ['identical %d/%d objects' % (len(same), len(a.keys() | b.keys()))]
    lines += [name + ': identical' for name in same]
    lines += different
    return not different, '\n'.join(lines) + '\n'


def functions(reference, candidate, member):
    """Diagnostic only: object parsing never enters the compiler pipeline."""
    import tempfile
    from objfile import Object
    archives = dict(read_archive(reference)), dict(read_archive(candidate))
    if any(member not in archive for archive in archives):
        return member + ': object missing\n'
    with tempfile.TemporaryDirectory() as where:
        objects = []
        for n, archive in enumerate(archives):
            path = Path(where) / ('%d.o' % n)
            path.write_bytes(archive[member])
            objects.append(Object(path))
        a, b = objects
        lines, same, total = [], 0, 0
        for s in a.symbols:
            if s.type != 'STT_FUNC':
                continue
            total += 1
            x, y = a.sections[s.section], b.sections.get(s.section)
            if y is not None and (x.data, x.relocs) == (y.data, y.relocs):
                same += 1
                lines.append('%s: identical code and relocations' % s.name)
            elif y is None:
                lines.append('%s: missing' % s.name)
            else:
                offset = next((i for i, (p, q) in enumerate(zip(x.data, y.data)) if p != q), min(len(x.data), len(y.data)))
                lines.append('%s: %d / %d bytes, first difference %#x%s' %
                             (s.name, len(x.data), len(y.data), offset, '; relocations differ' if x.relocs != y.relocs else ''))
        return '%s: identical code and relocations in %d/%d functions\n%s\n' % (member, same, total, '\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reference', type=Path)
    ap.add_argument('candidate', type=Path)
    ap.add_argument('--functions', metavar='MEMBER', help='also show function-level diagnostics for one archive member')
    args = ap.parse_args()
    ok, report = compare(args.reference, args.candidate)
    print(report, end='')
    if args.functions:
        print(functions(args.reference, args.candidate, args.functions), end='')
    return 0 if ok else 1
