from pathlib import Path
from argparse import ArgumentTypeError
from collections.abc import Iterable, Iterator

from pandas import DataFrame

type OptionallyLabeledSource = tuple[Path, str | None]
type LabeledSource = tuple[Path, str]
type LabeledData = tuple[str, DataFrame]


def existing_parent(arg: str) -> Path:
    p = Path(arg)
    if not p.parent.exists():
        raise ArgumentTypeError(f"parent directory does not exist: {arg}")
    return p


def existing_file(arg: str) -> Path:
    p = Path(arg)
    if not p.exists():
        raise ArgumentTypeError(f"file not found: {arg}")
    return p


def path_name_pair(arg: str) -> OptionallyLabeledSource:
    vals = arg.split("=")
    if len(vals) > 2:
        raise ArgumentTypeError(f"too many segments in path-name pair: {arg}")

    p = existing_file(vals[0])

    if not p.is_dir():
        raise ArgumentTypeError(f"file is not a directory: {arg}")

    if len(vals) == 1:
        return (p, None)

    label = vals[1].replace("_", " ").capitalize()
    return (p, label)


def default_labeled_source(
    sources: Iterable[OptionallyLabeledSource],
) -> Iterator[LabeledSource]:
    """Default unlabeled sources to use source stem."""
    for path, label in sources:
        yield path, label if label else path.stem
