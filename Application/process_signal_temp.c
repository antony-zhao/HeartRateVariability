#include <stdlib.h>
#include <stdbool.h>

void process_signal(float *sig, int *argmax, int argmax_len, int sig_len, float threshold,
                    float min_dist, float max_dist, float *average_interval, int avg_len,
                    int *processed_sig, bool *first, int *dist) {
    return;
    float average_sum = 0.0;

    for (int i = 0; i < avg_len; ++i) {
        average_sum += average_interval[i];
    }
    float average_mean = average_sum / avg_len;

    for (int i = 0; i < sig_len; ++i) {
        bool is_argmax = false;
        for (int j = 0; j < argmax_len; ++j) {
            if (i == argmax[j]) {
                is_argmax = true;
                break;
            }
        }

        if (is_argmax && sig[i] > threshold) {
            if (*dist < min_dist * average_mean) {
                if (*first) {
//                    processed_sig[i] = 1;
                    *first = false;
                } else {
//                    processed_sig[i] = 0;
                }
            } else {
//                processed_sig[i] = 1;
                if (*dist < max_dist * average_mean) {
                    // Update average_interval
                    average_sum -= average_interval[0];
                    for (int k = 1; k < avg_len; ++k) {
                        average_interval[k - 1] = average_interval[k];
                    }
                    average_interval[avg_len - 1] = *dist;
                    average_sum += *dist;
                    average_mean = average_sum / avg_len;
                }
                *dist = 0;
            }
        } else {
//            processed_sig[i] = 0;
        }
        *dist += 1.0;
    }
    free(sig);
    free(argmax);
}