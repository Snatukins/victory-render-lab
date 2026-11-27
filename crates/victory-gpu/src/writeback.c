#include <windows.h>
#include <stdint.h>

static int write_fault(EXCEPTION_POINTERS *info, const void *destination, size_t length)
{
    const EXCEPTION_RECORD *record = info->ExceptionRecord;
    const DWORD code = record->ExceptionCode;
    if ((code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_IN_PAGE_ERROR)
        && record->NumberParameters >= 2
        && record->ExceptionInformation[0] == 1
        && record->ExceptionInformation[1] - (uintptr_t)destination < length) {
        return EXCEPTION_EXECUTE_HANDLER;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

// The Rust caller cuts the range at system page boundaries. If this page
// disappears, a destination write can fail without losing the following page.
size_t victory_write_gpu_delta(
    volatile uint8_t *destination,
    const uint8_t *before,
    uint8_t *host,
    const uint8_t *title,
    size_t length)
{
    volatile size_t written = 0;
    __try {
        for (size_t i = 0; i < length; ++i) {
            const uint8_t gpu = host[i];
            if (gpu != before[i] && gpu != title[i]) {
                destination[i] = gpu;
                ++written;
            } else {
                host[i] = title[i];
            }
        }
    } __except (write_fault(GetExceptionInformation(), (const void *)destination, length)) {
        // The mapping changed during retirement. Leave the page dirty so
        // the next submission can take its current contents.
    }
    return written;
}
