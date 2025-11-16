#ifndef DEBUG_LOG_H
#define DEBUG_LOG_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the background debug log writer task. Returns true when
 * the task, queueing primitives, and buffers are ready to accept data. */
bool debug_log_init(void);

/* Enqueue a chunk of stdout/stderr text for persistence in the rolling log. */
void debug_log_capture(const char *data, size_t length);

#ifdef __cplusplus
}
#endif

#endif /* DEBUG_LOG_H */
