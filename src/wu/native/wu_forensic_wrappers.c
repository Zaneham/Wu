/**
 * Wu Forensics - H.264 Forensic Analysis Wrappers
 *
 * C wrappers for assembly-accelerated forensic functions.
 * Provides fallback implementations when assembly is unavailable.
 */

#include "wu_forensic.h"
#include <stdlib.h>
#include <string.h>

/* Forward declarations for assembly functions */
extern int wu_asm_qp_scan_horizontal(const int8_t*, int, int, int32_t*, int);
extern void wu_asm_qp_frame_stats(const int8_t*, int, int, int32_t*);
extern void wu_asm_qp_histogram(const int8_t*, int, int32_t*);
extern int wu_asm_qp_frame_discontinuities(const float*, int, float, int32_t*, int);

extern int wu_asm_mv_scan_horizontal(const int16_t*, int, int, int32_t*, int);
extern void wu_asm_mv_field_stats(const int16_t*, int, int, int32_t*, int);
extern int wu_asm_mv_coherence_score(const int16_t*, int, int);
extern int wu_asm_mv_detect_outliers(const int16_t*, int, int, int32_t*, int, int);

extern int wu_asm_scan_nal_units(const uint8_t*, int, uint8_t*, int32_t*, int);
extern int wu_asm_analyse_nal_sequence(const uint8_t*, int, int32_t*, int);
extern int wu_asm_count_epb(const uint8_t*, int);
extern void wu_asm_entropy_stats(const uint8_t*, int, int32_t*);

/* Flag indicating assembly availability (set by init) */
static int g_asm_available = 0;

/* Initialisation - call once at startup */
WU_EXPORT void wu_forensic_init(int use_asm) {
    g_asm_available = use_asm;
}


/* ============================================================================
 * QP Analysis - Fallback Implementations
 * ============================================================================ */

static int wu_qp_scan_horizontal_c(
    const int8_t* qp_row, int width_mbs, int threshold,
    int32_t* boundaries, int max_out
) {
    int count = 0;
    for (int i = 0; i < width_mbs - 1 && count < max_out; i++) {
        int delta = qp_row[i] - qp_row[i + 1];
        if (delta < 0) delta = -delta;
        if (delta >= threshold) {
            boundaries[count * 2] = i;
            boundaries[count * 2 + 1] = delta;
            count++;
        }
    }
    return count;
}

static void wu_qp_frame_stats_c(
    const int8_t* qp_map, int width_mbs, int height_mbs, int32_t* stats_out
) {
    int total = width_mbs * height_mbs;
    int sum = 0, min = 255, max = 0;

    for (int i = 0; i < total; i++) {
        int qp = (uint8_t)qp_map[i];
        sum += qp;
        if (qp < min) min = qp;
        if (qp > max) max = qp;
    }

    stats_out[0] = sum;
    stats_out[1] = min;
    stats_out[2] = max;
    stats_out[3] = total;
}

static void wu_qp_histogram_c(const int8_t* qp_map, int count, int32_t* histogram) {
    memset(histogram, 0, 52 * sizeof(int32_t));
    for (int i = 0; i < count; i++) {
        int qp = (uint8_t)qp_map[i];
        if (qp <= 51) histogram[qp]++;
    }
}

static int wu_qp_frame_discontinuities_c(
    const float* avg_qp_history, int n_frames, float threshold,
    int32_t* discontinuities, int max_out
) {
    int count = 0;
    for (int i = 1; i < n_frames && count < max_out; i++) {
        float delta = avg_qp_history[i] - avg_qp_history[i - 1];
        if (delta < 0) delta = -delta;
        if (delta >= threshold) {
            discontinuities[count++] = i;
        }
    }
    return count;
}


/* ============================================================================
 * MV Analysis - Fallback Implementations
 * ============================================================================ */

static int wu_mv_scan_horizontal_c(
    const int16_t* mv_row, int width_mbs, int threshold_sq,
    int32_t* discontinuities, int max_out
) {
    int count = 0;
    for (int i = 0; i < width_mbs - 1 && count < max_out; i++) {
        int dx = mv_row[i * 2] - mv_row[(i + 1) * 2];
        int dy = mv_row[i * 2 + 1] - mv_row[(i + 1) * 2 + 1];
        int mag_sq = dx * dx + dy * dy;
        if (mag_sq >= threshold_sq) {
            discontinuities[count * 2] = i;
            discontinuities[count * 2 + 1] = mag_sq;
            count++;
        }
    }
    return count;
}

static void wu_mv_field_stats_c(
    const int16_t* mv_field, int width_mbs, int height_mbs,
    int32_t* stats_out, int outlier_threshold
) {
    int total = width_mbs * height_mbs;
    int sum_mag_sq = 0, max_mag_sq = 0, outlier_count = 0;

    for (int i = 0; i < total; i++) {
        int mvx = mv_field[i * 2];
        int mvy = mv_field[i * 2 + 1];
        int mag_sq = mvx * mvx + mvy * mvy;
        sum_mag_sq += mag_sq;
        if (mag_sq > max_mag_sq) max_mag_sq = mag_sq;
        if (mag_sq >= outlier_threshold) outlier_count++;
    }

    stats_out[0] = sum_mag_sq;
    stats_out[1] = max_mag_sq;
    stats_out[2] = outlier_count;
    stats_out[3] = total;
}

static int wu_mv_coherence_score_c(
    const int16_t* mv_field, int width_mbs, int height_mbs
) {
    if (width_mbs < 3 || height_mbs < 3) return 100;

    int total_gradient = 0;
    int count = 0;
    int stride = width_mbs * 2;

    for (int y = 1; y < height_mbs - 1; y++) {
        for (int x = 1; x < width_mbs - 1; x++) {
            int idx = (y * width_mbs + x) * 2;
            int mvx = mv_field[idx];
            int mvy = mv_field[idx + 1];

            /* Gradient to 4 neighbours */
            int neighbours[4][2] = {
                {mv_field[idx - 2], mv_field[idx - 1]},         /* left */
                {mv_field[idx + 2], mv_field[idx + 3]},         /* right */
                {mv_field[idx - stride], mv_field[idx - stride + 1]},  /* top */
                {mv_field[idx + stride], mv_field[idx + stride + 1]}   /* bottom */
            };

            for (int n = 0; n < 4; n++) {
                int dx = mvx - neighbours[n][0];
                int dy = mvy - neighbours[n][1];
                total_gradient += dx * dx + dy * dy;
                count++;
            }
        }
    }

    if (count == 0) return 100;
    int avg = total_gradient / count;
    int normalised = avg / 100;
    if (normalised > 100) normalised = 100;
    return 100 - normalised;
}

static int wu_mv_detect_outliers_c(
    const int16_t* mv_field, int width_mbs, int height_mbs,
    int32_t* outliers, int threshold, int max_out
) {
    if (width_mbs < 3 || height_mbs < 3) return 0;

    int count = 0;
    int stride = width_mbs * 2;

    for (int y = 1; y < height_mbs - 1 && count < max_out; y++) {
        for (int x = 1; x < width_mbs - 1 && count < max_out; x++) {
            int idx = (y * width_mbs + x) * 2;
            int mvx = mv_field[idx];
            int mvy = mv_field[idx + 1];

            /* Average of 4 neighbours */
            int avg_x = (mv_field[idx - 2] + mv_field[idx + 2] +
                        mv_field[idx - stride] + mv_field[idx + stride]) / 4;
            int avg_y = (mv_field[idx - 1] + mv_field[idx + 3] +
                        mv_field[idx - stride + 1] + mv_field[idx + stride + 1]) / 4;

            int dx = mvx - avg_x;
            int dy = mvy - avg_y;
            int deviation_sq = dx * dx + dy * dy;

            if (deviation_sq >= threshold) {
                outliers[count * 3] = x;
                outliers[count * 3 + 1] = y;
                outliers[count * 3 + 2] = deviation_sq;
                count++;
            }
        }
    }
    return count;
}


/* ============================================================================
 * Bitstream Analysis - Fallback Implementations
 * ============================================================================ */

static int wu_scan_nal_units_c(
    const uint8_t* buffer, int length, uint8_t* nal_types,
    int32_t* nal_offsets, int max_nals
) {
    int count = 0;

    for (int i = 0; i < length - 4 && count < max_nals; i++) {
        /* Look for 00 00 01 or 00 00 00 01 */
        if (buffer[i] == 0 && buffer[i + 1] == 0) {
            int header_pos;
            if (buffer[i + 2] == 1) {
                header_pos = i + 3;
            } else if (buffer[i + 2] == 0 && buffer[i + 3] == 1) {
                header_pos = i + 4;
            } else {
                continue;
            }

            if (header_pos < length) {
                nal_types[count] = buffer[header_pos] & 0x1F;
                nal_offsets[count] = i;
                count++;
                i = header_pos;
            }
        }
    }
    return count;
}

static int wu_analyse_nal_sequence_c(
    const uint8_t* nal_types, int count, int32_t* anomalies, int max_anomalies
) {
    int anomaly_count = 0;
    int seen_sps = 0, seen_pps = 0, seen_slice = 0, seen_idr = 0;

    for (int i = 0; i < count && anomaly_count < max_anomalies; i++) {
        int nal_type = nal_types[i];

        /* Invalid NAL type */
        if (nal_type > 31) {
            anomalies[anomaly_count * 2] = i;
            anomalies[anomaly_count * 2 + 1] = 6;
            anomaly_count++;
            continue;
        }

        if (nal_type == 7) {  /* SPS */
            if (seen_slice && anomaly_count < max_anomalies) {
                anomalies[anomaly_count * 2] = i;
                anomalies[anomaly_count * 2 + 1] = 3;
                anomaly_count++;
            }
            seen_sps = 1;
        } else if (nal_type == 8) {  /* PPS */
            if (seen_slice && anomaly_count < max_anomalies) {
                anomalies[anomaly_count * 2] = i;
                anomalies[anomaly_count * 2 + 1] = 4;
                anomaly_count++;
            }
            seen_pps = 1;
        } else if (nal_type == 5) {  /* IDR */
            if (!seen_slice) {
                if (!seen_sps && anomaly_count < max_anomalies) {
                    anomalies[anomaly_count * 2] = i;
                    anomalies[anomaly_count * 2 + 1] = 1;
                    anomaly_count++;
                }
                if (!seen_pps && anomaly_count < max_anomalies) {
                    anomalies[anomaly_count * 2] = i;
                    anomalies[anomaly_count * 2 + 1] = 2;
                    anomaly_count++;
                }
            } else if (seen_idr && anomaly_count < max_anomalies) {
                anomalies[anomaly_count * 2] = i;
                anomalies[anomaly_count * 2 + 1] = 7;
                anomaly_count++;
            }
            seen_slice = 1;
            seen_idr = 1;
        } else if (nal_type == 1) {  /* Non-IDR */
            if (!seen_idr && anomaly_count < max_anomalies) {
                anomalies[anomaly_count * 2] = i;
                anomalies[anomaly_count * 2 + 1] = 5;
                anomaly_count++;
            }
            seen_slice = 1;
            seen_idr = 0;
        }
    }
    return anomaly_count;
}

static int wu_count_epb_c(const uint8_t* buffer, int length) {
    int count = 0;
    for (int i = 0; i < length - 2; i++) {
        if (buffer[i] == 0 && buffer[i + 1] == 0 && buffer[i + 2] == 3) {
            count++;
            i += 2;
        }
    }
    return count;
}

#ifdef _MSC_VER
static inline int popcount32(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}
#else
#define popcount32 __builtin_popcount
#endif

static void wu_entropy_stats_c(const uint8_t* buffer, int length, int32_t* stats) {
    int zero_count = 0, one_count = 0, byte_sum = 0, transitions = 0;

    for (int i = 0; i < length; i++) {
        uint8_t b = buffer[i];
        byte_sum += b;
        int ones = popcount32(b);
        one_count += ones;
        zero_count += 8 - ones;

        if (i > 0) {
            transitions += popcount32(b ^ buffer[i - 1]);
        }
    }

    stats[0] = zero_count;
    stats[1] = one_count;
    stats[2] = byte_sum;

    /* Entropy estimate */
    int denominator = length * 4;
    stats[3] = denominator ? (transitions * 100) / denominator : 0;
}


/* ============================================================================
 * Exported Wrapper Functions
 * ============================================================================ */

WU_EXPORT int wu_qp_scan_horizontal(
    const int8_t* qp_row, int width_mbs, int threshold,
    int32_t* boundaries, int max_out
) {
#ifdef WU_HAS_ASM
    if (g_asm_available) {
        return wu_asm_qp_scan_horizontal(qp_row, width_mbs, threshold, boundaries, max_out);
    }
#endif
    return wu_qp_scan_horizontal_c(qp_row, width_mbs, threshold, boundaries, max_out);
}

WU_EXPORT void wu_qp_frame_stats(
    const int8_t* qp_map, int width_mbs, int height_mbs, int32_t* stats_out
) {
#ifdef WU_HAS_ASM
    if (g_asm_available) {
        wu_asm_qp_frame_stats(qp_map, width_mbs, height_mbs, stats_out);
        return;
    }
#endif
    wu_qp_frame_stats_c(qp_map, width_mbs, height_mbs, stats_out);
}

WU_EXPORT void wu_qp_histogram(
    const int8_t* qp_map, int count, int32_t* histogram
) {
#ifdef WU_HAS_ASM
    if (g_asm_available) {
        wu_asm_qp_histogram(qp_map, count, histogram);
        return;
    }
#endif
    wu_qp_histogram_c(qp_map, count, histogram);
}

WU_EXPORT int wu_qp_frame_discontinuities(
    const float* avg_qp_history, int n_frames, float threshold,
    int32_t* discontinuities, int max_out
) {
#ifdef WU_HAS_ASM
    if (g_asm_available) {
        return wu_asm_qp_frame_discontinuities(avg_qp_history, n_frames, threshold,
                                               discontinuities, max_out);
    }
#endif
    return wu_qp_frame_discontinuities_c(avg_qp_history, n_frames, threshold,
                                         discontinuities, max_out);
}

WU_EXPORT int wu_mv_scan_horizontal(
    const int16_t* mv_row, int width_mbs, int threshold_sq,
    int32_t* discontinuities, int max_out
) {
#ifdef WU_HAS_ASM
    if (g_asm_available) {
        return wu_asm_mv_scan_horizontal(mv_row, width_mbs, threshold_sq,
                                         discontinuities, max_out);
    }
#endif
    return wu_mv_scan_horizontal_c(mv_row, width_mbs, threshold_sq,
                                   discontinuities, max_out);
}

WU_EXPORT void wu_mv_field_stats(
    const int16_t* mv_field, int width_mbs, int height_mbs,
    int32_t* stats_out, int outlier_threshold
) {
#ifdef WU_HAS_ASM
    if (g_asm_available) {
        wu_asm_mv_field_stats(mv_field, width_mbs, height_mbs, stats_out, outlier_threshold);
        return;
    }
#endif
    wu_mv_field_stats_c(mv_field, width_mbs, height_mbs, stats_out, outlier_threshold);
}

WU_EXPORT int wu_mv_coherence_score(
    const int16_t* mv_field, int width_mbs, int height_mbs
) {
#ifdef WU_HAS_ASM
    if (g_asm_available) {
        return wu_asm_mv_coherence_score(mv_field, width_mbs, height_mbs);
    }
#endif
    return wu_mv_coherence_score_c(mv_field, width_mbs, height_mbs);
}

WU_EXPORT int wu_mv_detect_outliers(
    const int16_t* mv_field, int width_mbs, int height_mbs,
    int32_t* outliers, int threshold, int max_out
) {
#ifdef WU_HAS_ASM
    if (g_asm_available) {
        return wu_asm_mv_detect_outliers(mv_field, width_mbs, height_mbs,
                                         outliers, threshold, max_out);
    }
#endif
    return wu_mv_detect_outliers_c(mv_field, width_mbs, height_mbs,
                                   outliers, threshold, max_out);
}

WU_EXPORT int wu_scan_nal_units(
    const uint8_t* buffer, int length, uint8_t* nal_types,
    int32_t* nal_offsets, int max_nals
) {
#ifdef WU_HAS_ASM
    if (g_asm_available) {
        return wu_asm_scan_nal_units(buffer, length, nal_types, nal_offsets, max_nals);
    }
#endif
    return wu_scan_nal_units_c(buffer, length, nal_types, nal_offsets, max_nals);
}

WU_EXPORT int wu_analyse_nal_sequence(
    const uint8_t* nal_types, int count, int32_t* anomalies, int max_anomalies
) {
#ifdef WU_HAS_ASM
    if (g_asm_available) {
        return wu_asm_analyse_nal_sequence(nal_types, count, anomalies, max_anomalies);
    }
#endif
    return wu_analyse_nal_sequence_c(nal_types, count, anomalies, max_anomalies);
}

WU_EXPORT int wu_count_epb(const uint8_t* buffer, int length) {
#ifdef WU_HAS_ASM
    if (g_asm_available) {
        return wu_asm_count_epb(buffer, length);
    }
#endif
    return wu_count_epb_c(buffer, length);
}

WU_EXPORT void wu_entropy_stats(const uint8_t* buffer, int length, int32_t* stats) {
#ifdef WU_HAS_ASM
    if (g_asm_available) {
        wu_asm_entropy_stats(buffer, length, stats);
        return;
    }
#endif
    wu_entropy_stats_c(buffer, length, stats);
}
