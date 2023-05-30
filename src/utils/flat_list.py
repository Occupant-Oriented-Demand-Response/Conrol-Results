from typing import Any


def flat_list(list_of_lists: list[list[Any]]) -> list[Any]:
    """Returns a flat list containing all elements from the sublists of the supplied list of lists."""
    flat_list = []
    for sublist in list_of_lists:
        flat_list.extend(sublist)
    return flat_list
