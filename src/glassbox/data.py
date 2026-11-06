"""
Data access: personas (shipped) and the German Credit data (downloaded).

The runtime only needs the shipped personas, the feature map and the model.
Downloading and pre-processing the UCI German Credit data -- needed for
retraining, the experiment notebooks and the data checks -- is delegated to
`MLaaS <https://github.com/So-Cool/MLaaS>`_, installed with the ``data`` extra
(``pip install "glassbox[data]"``).
"""
import dataclasses
import functools
import hashlib

from glassbox.resources import load_json

__all__ = ['Persona', 'load_personas', 'load_feature_metadata',
           'load_german_credit', 'processed_data_checksum',
           'verify_german_credit', 'PROCESSED_DATA_SHA256', 'FEATURE_SUBSET',
           'personas_matching_data', 'persona_names']

#: SHA-256 of the canonical CSV export (see :func:`processed_data_checksum`)
#: of the pre-processed data used to train the Glass-Box model.
PROCESSED_DATA_SHA256 = \
    '77f23ca45c8a1154ffab954d88766f55d098810208e42466613dc780f49d6f9a'
#: Features of the Glass-Box model (in the order of the original data subset).
FEATURE_SUBSET = list(load_json('german_credit_features.json')['feature_id'])
GERMAN_CREDIT_MODULE = 'mlaas.data.german_credit_simplified'


@dataclasses.dataclass
class Persona(object):
    """A loan applicant of the demo."""
    id: str
    name: str
    values: dict
    source: dict = None


def load_personas():
    """The ten demo personas (``id -> Persona``)."""
    return {p['id']: Persona(**p)
            for p in load_json('personas.json')['personas']}


@functools.lru_cache(maxsize=None)
def _persona_names():
    return tuple((i, p.name) for i, p in load_personas().items())


def persona_names():
    """``persona id -> name`` of the demo personas."""
    return dict(_persona_names())


def load_feature_metadata():
    """Feature ids, class id, feature map and one-hot columns of the model."""
    return load_json('german_credit_features.json')


def _mlaas():
    try:
        from mlaas.data_loader import DataHolder
    except ImportError as e:  # pragma: no cover
        raise ImportError('Data downloading requires MLaaS: '
                          'pip install "glassbox[data]".') from e
    return DataHolder


def load_german_credit(data_home=None, verify=True):
    """
    Downloads (if needed) and pre-processes the UCI German Credit data.

    Parameters
    ----------
    data_home : str, optional
        Download cache (default: ``~/.cache/mlaas`` or ``$MLAAS_DATA``).
    verify : bool
        Check that the pre-processed data match the data used in 2018.

    Returns
    -------
    mlaas.data_loader.DataHolder
        ``data`` (structured array), ``data_onehot`` (2-D array) and
        ``target`` of the 13-feature simplified data.
    """
    DataHolder = _mlaas()
    dh = DataHolder(feature_subset=FEATURE_SUBSET,
                    raw_data_module=GERMAN_CREDIT_MODULE,
                    data_home=data_home)
    if verify:
        verify_german_credit(dh)
    return dh


def processed_data_checksum(data):
    """SHA-256 of a canonical CSV export of a structured array."""
    lines = [','.join(data.dtype.names)]
    lines += [','.join(str(v) for v in row.tolist()) for row in data]
    return hashlib.sha256(('\n'.join(lines) + '\n').encode()).hexdigest()


def verify_german_credit(data_holder):
    """Raises ``ValueError`` if the pre-processed data differ from 2018."""
    checksum = processed_data_checksum(data_holder.data)
    if checksum != PROCESSED_DATA_SHA256:
        raise ValueError('The pre-processed German Credit data differ from '
                         'the data used by Glass-Box (checksum {}).'.format(
                             checksum))
    return True


def personas_matching_data(data_holder, personas=None):
    """
    Checks the personas copied from the data set against the data.

    Returns
    -------
    dict
        ``persona id -> True/False`` for every persona with a source row.
    """
    personas = load_personas() if personas is None else personas
    out = {}
    for p in personas.values():
        if not p.source:
            continue
        row = data_holder.data[p.source['row']]
        out[p.id] = all(str(row[f].item()) == str(v)
                        for f, v in p.values.items())
    return out
