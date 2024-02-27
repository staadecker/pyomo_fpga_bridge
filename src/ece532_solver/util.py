import time
from typing import Collection


def print_progress(
    collection: Collection, message, check_progress_every=5000, min_print_interval=0.4
):
    """Generator that given a collection will allow the caller to loop over and will
    print the progress in the meantime."""
    start_time = time.time()
    prev_print_time = 0
    total_len = len(collection)
    for i, val in enumerate(collection):
        if i % check_progress_every == 0:
            cur_time = time.time()
            if (cur_time - prev_print_time) > min_print_interval:
                print(f"{message} {(i/total_len):.1%}...", end="\r")
                prev_print_time = cur_time

        yield val
    print(f"{message}. Done in {(time.time() - start_time):.1f} s")
