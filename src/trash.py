import typing as tp


def _is_empty(x: tp.Any) -> bool:
    return x is None or x == ""


def _common_type(data: list[tp.Any]) -> type:
    types = {type(x) for x in data if not _is_empty(x)}
    if not types:
        return str
    if list in types or tuple in types:
        return list
    if str in types:
        return str
    if float in types:
        return float
    if bool in types:
        return bool
    return int


def _convert(x: tp.Any, target: type) -> tp.Any:
    if _is_empty(x):
        return target()
    if target is list:
        return list(x) if isinstance(x, (list, tuple)) else [x]
    return target(x)


def convert_to_common_type(data: list[tp.Any]) -> list[tp.Any]:
    """
    Takes list of multiple types' elements and convert each element to common type according to given rules
    :param data: list of multiple types' elements
    :return: list with elements converted to common type
    """
    target = _common_type(data)
    return [_convert(x, target) for x in data]
