# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as np
#
# def process_signal_cython(np.ndarray[float, ndim=1] sig, int sig_len, np.ndarray[int, ndim=1] argmax, int argmax_len,
#                           np.ndarray[int, ndim=1] average_interval, int average_interval_len, float threshold,
#                           float min_dist, float max_dist, bint first, int dist):
def process_signal_cython(float[:] sig, int sig_len, int[:] argmax, int argmax_len,
                          int[:] average_interval, int average_interval_len, float threshold,
                          float min_dist, float max_dist, bint first, int dist):
    cdef int i, s, j
    cdef float avg = 0
    cdef np.ndarray[int, ndim=1] processed_sig = np.zeros(sig_len, dtype=np.int32)
    cdef int curr_argmax = argmax[0]
    cdef int curr_ind = 1

    for i in range(average_interval_len):
        avg += average_interval[i]

    avg /= average_interval_len

    for i in range(sig_len):
        if i == curr_argmax and sig[
            i] > threshold:  # Minimum value of the signal before other checks. May need to adjust this value.
            if dist < min_dist * avg:
                if first:  # The very first signal
                    s = 1
                    first = 0
                    dist = 0
                else:
                    s = 0
            else:
                s = 1
                if dist < avg * max_dist:
                    avg *= average_interval_len
                    avg -= average_interval[0]
                    avg += dist
                    avg /= average_interval_len
                    # print(avg)

                    for j in range(1, average_interval_len):
                        average_interval[j - 1] = average_interval[j]
                    average_interval[average_interval_len - 1] = dist

                dist = 0
        else:
            s = 0

        processed_sig[i] = int(s)
        dist += 1
        if i >= curr_argmax:
            curr_argmax = argmax[curr_ind]
            curr_ind += 1

    del sig, argmax

    return processed_sig, average_interval, first, dist
