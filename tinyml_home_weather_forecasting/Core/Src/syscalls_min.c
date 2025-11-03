/*
 * ============================================================================
 *  @file    syscalls_min.c
 *  @brief   Minimal newlib syscall stubs for bare-metal STM32F767, no OS.
 *           Implements the small set newlib needs so the linker does not
 *           pull failing nosys stubs and emit warnings.
 *
 *  Barr-C style header
 *  Purpose:
 *      Provide minimal reentrant syscalls so printf/scanf and newlib internals
 *      link cleanly without semihosting or an actual filesystem/OS.
 *
 *  Params/Returns:
 *      Standard POSIX-like signatures; see each function.
 *
 *  Side effects:
 *      _write() can be wired to a UART or ITM if desired.
 *
 *  Pre/Postconditions:
 *      None. Works on bare-metal.
 *
 *  Concurrency:
 *      Reentrant forms (_*_r) are used by newlib-nano. No extra locking here.
 *
 *  Timing:
 *      _write() is blocking if you wire it to UART.
 *
 *  Errors:
 *      Sets errno to sensible values where applicable.
 *
 *  Notes:
 *      If you already have sysmem.c providing _sbrk(), keep it. This file does
 *      not implement _sbrk().
 * ============================================================================
 */

#include <sys/stat.h>
#include <sys/unistd.h>
#include <errno.h>
#include <stdint.h>

// Uncomment and implement if you want printf via UART or ITM.
// extern int UART_Debug_Transmit_Blocking(const uint8_t *buf, uint32_t len);

int _close(int file) {
    (void)file;
    errno = ENOSYS;
    return -1;
}

int _fstat(int file, struct stat *st) {
    (void)file;
    if (st) {
        st->st_mode = S_IFCHR;  // Claim character device for stdout/stderr
    }
    return 0;
}

int _isatty(int file) {
    // 0=stdin, 1=stdout, 2=stderr
    return (file == STDOUT_FILENO || file == STDERR_FILENO || file == STDIN_FILENO) ? 1 : 0;
}

int _lseek(int file, int ptr, int dir) {
    (void)file; (void)ptr; (void)dir;
    errno = ENOSYS;
    return -1;
}

int _read(int file, char *ptr, int len) {
    (void)file; (void)ptr; (void)len;
    // No stdin available; report EOF
    return 0;
}

int _getpid(void) {
    return 1;
}

int _kill(int pid, int sig) {
    (void)pid; (void)sig;
    errno = ENOSYS;
    return -1;
}
