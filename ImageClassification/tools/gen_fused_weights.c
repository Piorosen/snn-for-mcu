/*
 * Conv+AvgPool 퓨전 가중치 빌드타임 생성기 (호스트에서 실행).
 *
 * 5x5/pad2 conv 후 2x2/s2 avgpool은 "5x5 커널 4개(2x2 시프트)를 합친
 * 6x6 커널을 stride 2로 적용 후 /4"와 정수 수준에서 동일하다:
 *   w6[a][b] = sum_{dy,dx in {0,1}, 0<=a-dy<5, 0<=b-dx<5} w5[a-dy][b-dx]
 * (/4는 런타임 requantize 배율에 흡수)
 *
 * 합쳐진 값은 int8 4개의 합(±508)이라 int16으로 방출하고,
 * conv2가 144KB로 RAM에 못 올라가므로 FLASH const로 둔다.
 * FC 가중치의 HWC 퍼뮤트도 여기서 함께 방출해 런타임 RAM 사본을 없앤다.
 *
 * 실행 (snn_weights.c 변경 시 재실행):
 *   cd ImageClassification
 *   clang -O2 -I Core/Inc -o /tmp/genfw tools/gen_fused_weights.c && /tmp/genfw
 *   (Core/Src/snn_weights_fused.c 와 Core/Inc/snn_weights_fused.h 갱신)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "../Core/Src/snn_weights.c"

static void emit_i16(FILE *f, const char *name, const int16_t *v, int n) {
    fprintf(f, "const int16_t %s[%d] __attribute__((aligned(4))) = {\n", name, n);
    for (int i = 0; i < n; ++i)
        fprintf(f, "%d,%s", v[i], (i % 16 == 15) ? "\n" : " ");
    fprintf(f, "};\n\n");
}

static void emit_i8(FILE *f, const char *name, const int8_t *v, int n) {
    fprintf(f, "const int8_t %s[%d] = {\n", name, n);
    for (int i = 0; i < n; ++i)
        fprintf(f, "%d,%s", v[i], (i % 16 == 15) ? "\n" : " ");
    fprintf(f, "};\n\n");
}

/* [Co][Ci][5][5] -> 퓨전 [Co][6][6][Ci] (NHWC-호환, int16) */
static void fuse(const int8_t *w5, int Co, int Ci, int16_t *w6) {
    for (int co = 0; co < Co; ++co)
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b)
                for (int ci = 0; ci < Ci; ++ci) {
                    int v = 0;
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            int ka = a - dy, kb = b - dx;
                            if (ka >= 0 && ka < 5 && kb >= 0 && kb < 5)
                                v += w5[((co * Ci + ci) * 5 + ka) * 5 + kb];
                        }
                    w6[((co * 6 + a) * 6 + b) * Ci + ci] = (int16_t)v;
                }
}

int main(void) {
    static int16_t w1[32 * 6 * 6 * 3];
    static int16_t w2[64 * 6 * 6 * 32];
    static int8_t fc[10 * 4096];

    fuse((const int8_t *)snn_conv1_weight, 32, 3, w1);
    fuse((const int8_t *)snn_conv2_weight, 64, 32, w2);

    /* FC: 입력 flatten CHW(c*64+h*8+w) -> HWC((h*8+w)*64+c) */
    for (int o = 0; o < 10; ++o)
        for (int c = 0; c < 64; ++c)
            for (int h = 0; h < 8; ++h)
                for (int w = 0; w < 8; ++w)
                    fc[o * 4096 + (h * 8 + w) * 64 + c] =
                        snn_fc_weight[o][c * 64 + h * 8 + w];

    FILE *f = fopen("Core/Src/snn_weights_fused.c", "w");
    if (!f) { perror("snn_weights_fused.c"); return 1; }
    fprintf(f, "/* tools/gen_fused_weights.c 가 생성한 파일 — 직접 수정 금지.\n"
               " * snn_weights.c 변경 시 생성기를 재실행할 것. */\n"
               "#include \"snn_weights_fused.h\"\n\n");
    emit_i16(f, "snn_conv1_fused_w", w1, 32 * 6 * 6 * 3);
    emit_i16(f, "snn_conv2_fused_w", w2, 64 * 6 * 6 * 32);
    emit_i8(f, "snn_fc_weight_hwc", fc, 10 * 4096);
    fclose(f);

    f = fopen("Core/Inc/snn_weights_fused.h", "w");
    if (!f) { perror("snn_weights_fused.h"); return 1; }
    fprintf(f,
        "/* tools/gen_fused_weights.c 가 생성한 파일 — 직접 수정 금지. */\n"
        "#ifndef SNN_WEIGHTS_FUSED_H\n#define SNN_WEIGHTS_FUSED_H\n\n"
        "#include <stdint.h>\n\n"
        "/* Conv+AvgPool 퓨전 6x6/s2 커널, [Co][6][6][Ci], 5x5 커널 4개의 합.\n"
        " * 실배율 = (원본 conv 배율) / 4 — requantize에서 처리 */\n"
        "extern const int16_t snn_conv1_fused_w[32*6*6*3];\n"
        "extern const int16_t snn_conv2_fused_w[64*6*6*32];\n\n"
        "/* FC 가중치, 입력 HWC flatten 순서 [10][4096] */\n"
        "extern const int8_t snn_fc_weight_hwc[10*4096];\n\n"
        "#endif\n");
    fclose(f);

    printf("generated: snn_weights_fused.c/.h (conv1 %zuB, conv2 %zuB, fc %zuB)\n",
           sizeof(w1), sizeof(w2), sizeof(fc));
    return 0;
}
