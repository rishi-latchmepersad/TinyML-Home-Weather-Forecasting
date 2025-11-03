#include <stddef.h>   // size_t

/* Real libc allocation symbols that the linker will resolve for us */
void* __real_malloc(size_t size);
void  __real_free(void* ptr);

/* Wrapped symbols that satisfy -Wl,--wrap=malloc/free */
void* __wrap_malloc(size_t size) {
    return __real_malloc(size);
}

void __wrap_free(void* ptr) {
    __real_free(ptr);
}
