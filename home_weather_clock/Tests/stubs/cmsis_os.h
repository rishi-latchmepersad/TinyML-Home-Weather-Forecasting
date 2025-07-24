#ifndef __CMSIS_OS_H__
#define __CMSIS_OS_H__

typedef void* osThreadId;
typedef void (*os_pthread)(void const *argument);

typedef struct {
    os_pthread pthread;
    void *dummy; // add what you need
} osThreadDef_t;

#define osThreadDef(name, priority, instances, stacksize)  osThreadDef_t name = {0}
#define osThread(name) 0

#endif
