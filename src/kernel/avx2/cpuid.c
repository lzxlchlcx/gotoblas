#include <stdio.h>

#ifdef __GNUC__
#include <cpuid.h>
#endif

int cpu_supports_avx2(void)
{
#ifdef __GNUC__
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        int has_avx2 = (ebx & (1 << 5)) != 0;
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
        int has_fma = (ecx & (1 << 12)) != 0;
        int has_osxsave = (ecx & (1 << 27)) != 0;
        if (has_avx2 && has_fma && has_osxsave) {
            unsigned long long xcr0;
            __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
            xcr0 = ((unsigned long long)edx << 32) | eax;
            if ((xcr0 & 0x6) == 0x6) {
                return 1;
            }
        }
    }
#endif
    return 0;
}
