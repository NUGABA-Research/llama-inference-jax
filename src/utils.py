from typing import Any

def _as_int_list(x: Any) -> tuple[int, ...]:
    if x is None:
        return tuple()
    
    if isinstance(x, int):
        return (x,)
    
    if isinstance(x, (list, tuple)):
        return tuple(int(v) for v in x)
    
    raise TypeError(f"Expected int or list/tuple of ints, got: {type(x)}")

def _flatten_ids(ids: Any) -> list[int]:
    if isinstance(ids, list) and (len(ids) == 0 or isinstance(ids[0], int)):
        return list(ids)
    
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        return list(ids[0])
    
    raise TypeError(f"Unexpected token ids shape/type from tokenizer: {type(ids)}")