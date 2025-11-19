#include <stdio.h>
#include <vector>
#include <filesystem>   // C++17
#include <fstream>
#include <cctype>       // isdigit
#include <string>
#include <snn.h>

/* ---- 내부 버퍼 및 LIF 상태 정의 ---- */

bool load_cifar_bin(const std::string& path, float x[3][32][32])
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }

    // 파일 크기 체크 (선택사항)
    // float 3*32*32 = 3*32*32*4 bytes = 12288 bytes
    const std::size_t expected_bytes = 3 * 32 * 32 * sizeof(float);

    // 파일 내용을 한 번에 읽기
    ifs.read(reinterpret_cast<char*>(x), expected_bytes);

    if (!ifs) {
        // 읽기 실패 시 0으로 초기화
        // memset(x, 0, 3 * 32 * 32 * sizeof(float));
        return false;
    }
    return true;
}

int parse_label_from_filename(const std::string& filename)
{
    // 예: "0003_0123.bin" -> 앞 4자리가 label
    // 경로가 들어올 수도 있으니, 마지막 '/' 또는 '\\' 뒤부터 사용
    std::string name = filename;
    auto pos1 = name.find_last_of("/\\");
    if (pos1 != std::string::npos) {
        name = name.substr(pos1 + 1);
    }

    if (name.size() < 4) {
        return -1;
    }

    // 앞 4자리 숫자인지 확인
    for (int i = 0; i < 4; ++i) {
        if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
            return -1;
        }
    }

    // stoi로 label 추출
    try {
        int label = std::stoi(name.substr(0, 4));
        return label;
    } catch (...) {
        return -1;
    }
}


namespace fs = std::filesystem;

int main()
{
    std::string dir = "./cifar_bin";

    float x[3][32][32];
    float spikes[30][3][32][32];
    float spk_out[10];
    float mem_out[10];
    int total_samples  = 0;
    int correct_samples = 0;


    snn_reset_state();

    // 디렉토리 안의 .bin 파일 순회
    for (const auto& entry : fs::directory_iterator(dir)) {
        float calc[10] { 0, };
        if (!entry.is_regular_file()) continue;
        auto path = entry.path();
        if (path.extension() != ".bin") continue;

        std::string filepath = path.string();
        std::string filename = path.filename().string();

        int label = parse_label_from_filename(filename);
        if (!load_cifar_bin(filepath, x)) {
            continue;
        }
        spiking_rate(
            (const float*)x,
            (float*)spikes,  // 임시로 spikes에 스파이크 저장
            30, 1, 3, 32, 32,
            1, 0, 0
        );

        // 한 타임스텝 forward
        snn_reset_state();

        for (int t = 0; t < 30; ++t) {
            // spikes[t] -> [3][32][32] 라고 가정
            snn_forward_step((const float (*)[32][32])spikes[t], spk_out, mem_out);

            // 이번 타임스텝 스파이크를 calc에 누적
            for (int i = 0; i < 10; ++i) {
                calc[i] += spk_out[i];
            }
        }

        int pred = 0;
        float max_val = calc[0];
        for (int i = 1; i < 10; ++i) {
            if (calc[i] > max_val) {
                max_val = calc[i];
                pred = i;
            }
        }

        printf("Final calc = [");
        for (int i = 0; i < 10; ++i) {
            printf("%s%.3f", (i == 0 ? "" : ", "), calc[i]);
        }
        printf("]\n");
        printf("Prediction: answer = %d, class = %d, max_val = %.3f\n\n",
               label, pred, max_val);

        // ---- 정확도 집계 ----
        ++total_samples;
        if (pred == label) {
            ++correct_samples;
        }
    }

    // ---- 전체 정확도 출력 ----
    if (total_samples > 0) {
        double acc = (double)correct_samples / (double)total_samples * 100.0;
        printf("Total samples:   %d\n", total_samples);
        printf("Correct samples: %d\n", correct_samples);
        printf("Accuracy:        %.2f%%\n", acc);
    } else {
        printf("No valid samples processed.\n");
    }

    return 0;
}
