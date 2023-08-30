try:
    from collections.abc import MutableMapping
except:
    from collections import MutableMapping

# See reference: https://stackoverflow.com/questions/7760916/correct-usage-of-a-getter-setter-for-dictionary-values


from collections.abc import Mapping, MutableMapping
from functools import partial
import json


class Meta(MutableMapping):
    def __init__(self, meta={}):
        self._meta = meta

    def clear(self):
        return self._meta.clear()

    def __contains__(self, key):
        return key in self._meta

    def __setitem__(self, key, formula):
        self._meta[key] = formula

    def __getitem__(self, key):
        return self._meta[key]

    def __len__(self):
        return len(self._meta)

    def __iter__(self):
        return iter(self._meta)

    def __delitem__(self, key):
        del self._meta[key]

    def getformula(self, key):
        """Return raw un-evaluated contents of cell."""
        return self._meta[key]

    # def update(self, *args, **kwargs):
    #     for k, v in dict(*args, **kwargs).iteritems():
    #         self[k] = v

    def update(self, dict):
        for k, v in dict.items():
            self[k] = v

    def to_dict(self):
        return dict(self)

    def __str__(self) -> str:
        attrs = []
        for k, v in self._meta.items():
            attrs += [f"{k}:{v}"]
        desc = " ".join(attrs)
        return f"Metadata: {desc}"
