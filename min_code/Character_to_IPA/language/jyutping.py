"""
This file is copied from https://github.com/CanCLID/ToJyutping
and modified to fit our needs.
这个文件来自 https://github.com/CanCLID/ToJyutping
并修改了部分代码以适应我们的需求。
"""

import sys

from typing import Literal, Tuple, Union, overload

import re
import warnings
from itertools import starmap
from dataclasses import dataclass
from functools import cached_property
from operator import add, attrgetter


def is_iterable(o):
    try:
        iter(o)
    except TypeError:
        return False
    return True


onset = [
    "",
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "ng",
    "gw",
    "kw",
    "w",
    "h",
    "z",
    "c",
    "s",
    "j",
]
nucleus = ["aa", "a", "e", "i", "o", "u"]
rhyme = ["oe", "oen", "oeng", "oet", "oek", "eoi", "eon", "eot", "yu", "yun", "yut", "m", "ng"]
coda = ["", "i", "u", "m", "n", "ng", "p", "t", "k"]

regex = re.compile(
    "^([gk]w?|ng|[bpmfdtnlhwzcsj]?)(?![1-6]?$)((aa?|oe?|eo?|y?u|i?)(ng|[iumnptk]?))([1-6]?)$"
)

with_numeric_tone_regex = re.compile(r"^([^0-9]+)([0-9]+)$")

# 语保 香港
onset_ipa = [
    "",
    "p",
    "ph",
    "m",
    "f",
    "t",
    "th",
    "n",
    "l",
    "k",
    "kh",
    "ŋ",
    "ku",
    "khu",
    "u",
    "h",
    "ts",
    "tsh",
    "s",
    "i",
]

__our_flavor = set()
for __shengmu in onset_ipa:
    if not len(__shengmu.rstrip("iu")) == 0:
        __our_flavor.add(__shengmu.rstrip("iu"))
__yubao_xianggang = eval(
    "{'ts', 'k', 'Ǿ', 'h', 'tsh', 'p', 'l', 'ŋ', 'm', 't', 'kh', 's', 'f', 'th', 'n', 'ph'}"
)
assert (
    len(__our_flavor - __yubao_xianggang) == 0
), f"额外包含: {repr(__our_flavor - __yubao_xianggang)}"
print(f"未包含: {repr(__yubao_xianggang - __our_flavor)}", file=sys.stderr)

# 语保 香港
nucleus_ipa = ["a", "ɐ", "ɛ", "i", "ɔ", "u", "œ", "ø", "y"]
coda_ipa = ["", "i", "u", "m", "n", "ŋ", "p", "t", "k"]
tone_ipa = ["5", "35", "3", "21", "13", "2"]
# tone_ipa = ["˥", "˧˥", "˧", "˨˩", "˩˧", "˨"]

# 语保 香港
nucleus_map = dict(zip([*nucleus, "oe", "eo", "yu"], nucleus_ipa))
coda_map = dict(zip(coda, coda_ipa))
rhyme_ipa = {
    **{i: nucleus_map[r[:2]] + coda_map[r[2:]] for i, r in enumerate(rhyme[:-2], 54)},
    19: "ei",
    32: "eŋ",
    35: "ek",
    38: "ou",
    50: "oŋ",
    53: "ok",
    59: "øy",
    65: "m",
    66: "ŋ",
}

__our_flavor = set()
for __yunmu in rhyme_ipa.values():
    if __yunmu in ["m", "n", "ŋ"]:
        continue
    if __yunmu in ["œm", "œn", "œŋ", "œp", "œt", "œk"]:
        continue
    __our_flavor.add(__yunmu)
__yubao_xianggang = eval(
    "{'ɔk', 'ɐŋ', 'ɔi', 'iok', 'ueŋ', 'iœk', 'uɔk', 'uai', 'in', 'au', 'œŋ', 'ieŋ', 'ak', 'iɐm', 'i', 'uɐn', 'an', 'ɐk', 'a', 'ŋ', 'un', 'øn', 'ei', 'ɔŋ', 'iu', 'ioŋ', 'ek', 'ɔt', 'iɐn', 'uɔ', 'ɛ', 'ui', 'ai', 'ut', 'iɐt', 'uɐi', 'yn', 'ɔ', 'ɛk', 'ɛŋ', 'am', 'iøn', 'ɐi', 'ua', 'uan', 'uak', 'uɐt', 'œ', 'ɔn', 'ɐt', 'iɛŋ', 'ɐp', 'iek', 'y', 'uɔŋ', 'œk', 'ɐn', 'im', 'at', 'ou', 'ok', 'yt', 'iœŋ', 'u', 'ip', 'iɛ', 'øy', 'it', 'ap', 'oŋ', 'eŋ', 'øt', 'uat', 'ɐu', 'aŋ', 'uaŋ', 'ɐm', 'iɐu', 'iɐp'}"
)
assert (
    len(__our_flavor - __yubao_xianggang) == 0
), f"额外包含: {repr(__our_flavor - __yubao_xianggang)}"
print(f"未包含: {repr(__yubao_xianggang - __our_flavor)}", file=sys.stderr)

_minimal_mapping_nucleus_map = {"oe": 26, "eo": 26, "yu": 27}
_minimal_mapping_nucleus_to_onset = {3: 19, 5: 14}
_minimal_mapping_coda_to_onset = [0, 19, 14, 3, 7, 11, 1, 5, 9]
_minimal_mapping_rhyme_to_nucleus = {
    **{i: _minimal_mapping_nucleus_map[r[:2]] for i, r in enumerate(rhyme[:-2], 54)},
    19: 23,
    32: 23,
    35: 23,
    38: 25,
    50: 25,
    53: 25,
    65: 0,
    66: 0,
}
_minimal_mapping_rhyme_to_coda = {
    **{i: _minimal_mapping_coda_to_onset[coda.index(r[2:])] for i, r in enumerate(rhyme[:-2], 54)},
    0: 20,
    9: 21,
    18: 22,
    27: 19,
    36: 24,
    45: 14,
    54: 26,
    62: 27,
    65: 3,
    66: 11,
}


@dataclass(frozen=True)
class Jyutping:
    id: int
    onset_id: int
    onset: str
    rhyme_id: int
    rhyme: str
    tone_id: int
    tone: str
    jyutping: str

    def __init__(self, x: Union[str, int]):
        if type(x) == int:
            object.__setattr__(self, "id", x)
            object.__setattr__(self, "onset_id", x // 402)
            object.__setattr__(self, "onset", onset[self.onset_id])
            object.__setattr__(self, "rhyme_id", (x % 402) // 6)
            object.__setattr__(
                self,
                "rhyme",
                (
                    rhyme[self.rhyme_id - 54]
                    if self.rhyme_id >= 54
                    else nucleus[self.rhyme_id // 9] + coda[self.rhyme_id % 9]
                ),
            )
            object.__setattr__(self, "tone_id", x % 6)
            object.__setattr__(self, "tone", str(self.tone_id + 1))
            object.__setattr__(self, "jyutping", self.onset + self.rhyme + self.tone)
        else:
            object.__setattr__(self, "jyutping", x)
            match = re.match(regex, x)
            if not match:
                raise ValueError(f"Invalid jyutping: {x!r}")
            _onset, _rhyme, _nucleus, _coda, _tone = match.groups()
            object.__setattr__(self, "onset", _onset)
            object.__setattr__(self, "onset_id", onset.index(_onset))
            object.__setattr__(self, "rhyme", _rhyme)
            try:
                object.__setattr__(self, "rhyme_id", rhyme.index(_rhyme) + 54)
            except ValueError:
                object.__setattr__(
                    self, "rhyme_id", coda.index(_coda) + nucleus.index(_nucleus) * 9
                )
            object.__setattr__(self, "tone", _tone)
            object.__setattr__(self, "tone_id", int(_tone) - 1)
            object.__setattr__(self, "id", self.tone_id + self.rhyme_id * 6 + self.onset_id * 402)

    def __str__(self):
        return self.jyutping

    def __eq__(self, other):
        return isinstance(other, Jyutping) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @cached_property
    def ipa(self) -> str:
        return (
            onset_ipa[self.onset_id]
            + rhyme_ipa.get(
                self.rhyme_id, nucleus_ipa[self.rhyme_id // 9] + coda_ipa[self.rhyme_id % 9]
            )
            + tone_ipa[self.tone_id]
        )

    @overload
    def g2p(
        self,
        offset: Union[int, Tuple[int, int, int]] = 0,
        *,
        tone_same_seq=False,
        minimal: Literal[False] = False,
    ) -> Tuple[int, int, int]: ...

    @overload
    def g2p(
        self,
        offset: Union[int, Tuple[int, int, int, int]] = 0,
        *,
        tone_same_seq=False,
        minimal: Literal[True],
    ) -> Tuple[int, int, int, int]: ...

    def g2p(
        self,
        offset: Union[int, Tuple[int, int, int], Tuple[int, int, int, int]] = 0,
        *,
        tone_same_seq=False,
        minimal=False,
    ) -> Union[Tuple[int, int, int], Tuple[int, int, int, int]]:
        if minimal:
            warnings.warn(
                "`minimal` is an experimental feature and is subject to changes or removal in the future."
            )
            result = (
                self.onset_id,
                _minimal_mapping_rhyme_to_nucleus.get(
                    self.rhyme_id,
                    _minimal_mapping_nucleus_to_onset.get(
                        self.rhyme_id // 9, self.rhyme_id // 9 + 20
                    ),
                ),
                _minimal_mapping_rhyme_to_coda.get(
                    self.rhyme_id, _minimal_mapping_coda_to_onset[self.rhyme_id % 9]
                ),
                self.tone_id + (28 if tone_same_seq else 1),
            )
        else:
            result = (
                self.onset_id,
                self.rhyme_id + 20,
                self.tone_id + (87 if tone_same_seq else 1),
            )
        return (
            result
            if not offset
            else tuple(
                starmap(add, zip(result, offset))
                if is_iterable(offset)
                else map(offset.__add__, result)
            )
        )


def jyutping_to_ipa(jyutping: str) -> str:
    ipa_str = Jyutping(jyutping).ipa
    m = with_numeric_tone_regex.match(ipa_str)
    assert m, f"Invalid ipa: {repr(ipa_str)}"
    base, tone = m.group(1), m.group(2)
    if not base.endswith(("p", "t", "k")) and len(tone) == 1:
        tone = tone * 2
    return base + tone


# if __name__ == "__main__":
#     print(jyutping_to_ipa("bui1"))
#     print(jyutping_to_ipa("bit6"))
#     print(jyutping_to_ipa("co1"))
#     print(jyutping_to_ipa("bong1"))
#     print(jyutping_to_ipa("ng5"))
