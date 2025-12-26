/**
 * Wu Forensics - H.264 Forensic Analysis Header
 *
 * Declarations for assembly-accelerated forensic analysis functions.
 * These functions detect tampering indicators in H.264 video streams.
 */

#ifndef WU_FORENSIC_H
#define WU_FORENSIC_H

#include "wu_simd.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * QP (Quantisation Parameter) Analysis
 * ============================================================================ */

/**
 * Scan a row of QP values for sharp horizontal boundaries.
 * Sharp QP boundaries may indicate video splicing or compositing.
 *
 * @param qp_row       Array of QP values (int8_t), one per macroblock
 * @param width_mbs    Number of macroblocks in the row
 * @param threshold    Minimum delta to flag as boundary (typically 8-12)
 * @param boundaries   Output array of [mb_x, delta] pairs (int32_t)
 * @param max_out      Maximum number of boundaries to record
 * @return             Number of boundaries found
 */
WU_EXPORT int wu_qp_scan_horizontal(
    const int8_t* qp_row,
    int width_mbs,
    int threshold,
    int32_t* boundaries,
    int max_out
);

/**
 * Compute QP statistics for an entire frame.
 *
 * @param qp_map       QP values for all macroblocks (row-major)
 * @param width_mbs    Frame width in macroblocks
 * @param height_mbs   Frame height in macroblocks
 * @param stats_out    Output: [sum, min, max, count] as int32_t[4]
 */
WU_EXPORT void wu_qp_frame_stats(
    const int8_t* qp_map,
    int width_mbs,
    int height_mbs,
    int32_t* stats_out
);

/**
 * Build histogram of QP values (52 bins, 0-51).
 *
 * @param qp_map       QP values for all macroblocks
 * @param count        Total number of macroblocks
 * @param histogram    Output: int32_t[52] histogram
 */
WU_EXPORT void wu_qp_histogram(
    const int8_t* qp_map,
    int count,
    int32_t* histogram
);

/**
 * Detect frames with large QP changes from previous frame.
 * Used for splice detection at frame boundaries.
 *
 * @param avg_qp_history    Array of average QP per frame (float)
 * @param n_frames          Number of frames
 * @param threshold         Minimum delta to flag (typically 10-15)
 * @param discontinuities   Output: array of frame indices
 * @param max_out           Maximum discontinuities to record
 * @return                  Number of discontinuities found
 */
WU_EXPORT int wu_qp_frame_discontinuities(
    const float* avg_qp_history,
    int n_frames,
    float threshold,
    int32_t* discontinuities,
    int max_out
);


/* ============================================================================
 * Motion Vector Field Analysis
 * ============================================================================ */

/**
 * Scan a row of motion vectors for sharp spatial discontinuities.
 * MVs are stored as packed (mvx, mvy) int16_t pairs.
 *
 * @param mv_row           Array of MVs: [mvx0, mvy0, mvx1, mvy1, ...]
 * @param width_mbs        Number of macroblocks in row
 * @param threshold_sq     Squared magnitude threshold (e.g. 1024 = 32^2)
 * @param discontinuities  Output: [mb_x, magnitude_sq] pairs
 * @param max_out          Maximum discontinuities to record
 * @return                 Number of discontinuities found
 */
WU_EXPORT int wu_mv_scan_horizontal(
    const int16_t* mv_row,
    int width_mbs,
    int threshold_sq,
    int32_t* discontinuities,
    int max_out
);

/**
 * Compute MV field statistics.
 *
 * @param mv_field         All MVs as [mvx, mvy] pairs (row-major)
 * @param width_mbs        Frame width in macroblocks
 * @param height_mbs       Frame height in macroblocks
 * @param stats_out        Output: [sum_mag_sq, max_mag_sq, outlier_count, total]
 * @param outlier_threshold  Magnitude squared threshold for outlier
 */
WU_EXPORT void wu_mv_field_stats(
    const int16_t* mv_field,
    int width_mbs,
    int height_mbs,
    int32_t* stats_out,
    int outlier_threshold
);

/**
 * Compute MV field coherence score.
 * Coherent fields (natural video) have smooth MV gradients.
 *
 * @param mv_field      All MVs as [mvx, mvy] pairs
 * @param width_mbs     Frame width in macroblocks
 * @param height_mbs    Frame height in macroblocks
 * @return              Coherence score (0-100, higher = more coherent)
 */
WU_EXPORT int wu_mv_coherence_score(
    const int16_t* mv_field,
    int width_mbs,
    int height_mbs
);

/**
 * Detect MV outliers that deviate from local neighbourhood.
 *
 * @param mv_field      All MVs as [mvx, mvy] pairs
 * @param width_mbs     Frame width in macroblocks
 * @param height_mbs    Frame height in macroblocks
 * @param outliers      Output: [mb_x, mb_y, deviation_sq] triples
 * @param threshold     Deviation squared threshold (e.g. 2500 = 50^2)
 * @param max_out       Maximum outliers to record
 * @return              Number of outliers found
 */
WU_EXPORT int wu_mv_detect_outliers(
    const int16_t* mv_field,
    int width_mbs,
    int height_mbs,
    int32_t* outliers,
    int threshold,
    int max_out
);


/* ============================================================================
 * Bitstream Analysis
 * ============================================================================ */

/**
 * Scan buffer for NAL start codes and extract NAL type sequence.
 *
 * @param buffer       H.264 bitstream data
 * @param length       Buffer length in bytes
 * @param nal_types    Output: sequence of NAL types (uint8_t)
 * @param nal_offsets  Output: byte offset of each NAL (int32_t)
 * @param max_nals     Maximum NALs to record
 * @return             Number of NALs found
 */
WU_EXPORT int wu_scan_nal_units(
    const uint8_t* buffer,
    int length,
    uint8_t* nal_types,
    int32_t* nal_offsets,
    int max_nals
);

/**
 * Analyse NAL type sequence for structural anomalies.
 *
 * Anomaly codes:
 *   1 = Missing SPS before first slice
 *   2 = Missing PPS before first slice
 *   3 = SPS after slice (potential splice)
 *   4 = PPS after slice (potential splice)
 *   5 = Non-IDR slice without preceding IDR
 *   6 = Invalid NAL type (>31)
 *   7 = Consecutive IDRs (unusual)
 *
 * @param nal_types     Sequence of NAL types
 * @param count         Number of NALs
 * @param anomalies     Output: [index, anomaly_code] pairs
 * @param max_anomalies Maximum anomalies to record
 * @return              Number of anomalies found
 */
WU_EXPORT int wu_analyse_nal_sequence(
    const uint8_t* nal_types,
    int count,
    int32_t* anomalies,
    int max_anomalies
);

/**
 * Count emulation prevention bytes (00 00 03 sequences).
 * Unusual EPB patterns can indicate manipulation.
 *
 * @param buffer    NAL data
 * @param length    Buffer length
 * @return          Number of EPB sequences found
 */
WU_EXPORT int wu_count_epb(
    const uint8_t* buffer,
    int length
);

/**
 * Compute entropy statistics on slice data.
 *
 * @param buffer    Slice data after header
 * @param length    Buffer length
 * @param stats     Output: [zero_count, one_count, byte_sum, entropy_est]
 */
WU_EXPORT void wu_entropy_stats(
    const uint8_t* buffer,
    int length,
    int32_t* stats
);


#ifdef __cplusplus
}
#endif

#endif /* WU_FORENSIC_H */
